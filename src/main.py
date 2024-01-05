# Copyright (c) farm-ng, inc. Amiga Development Kit License, Version 0.1
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Literal
import cv2
from cv2.ximgproc import anisotropicDiffusion
import numpy as np
from turbojpeg import TurboJPEG
# import tensorflow as tf

from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.canbus.packet import AmigaControlState
from farm_ng.canbus.packet import AmigaTpdo1
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.events_file_reader import payload_to_protobuf
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri

# from kornia.utils import tensor_to_image
from virtual_joystick.joystick import VirtualJoystickWidget

# import internal libs

# Must come before kivy imports
os.environ["KIVY_NO_ARGS"] = "1"

# gui configs must go before any other kivy import
from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

# kivy imports
from kivy.app import App  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.properties import StringProperty  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.input.providers.mouse import MouseMotionEvent  # noqa: E402


logger = logging.getLogger("amiga.apps.camera")

MAX_LINEAR_VELOCITY_MPS = 0.5
MAX_ANGULAR_VELOCITY_RPS = 0.5
VELOCITY_INCREMENT = 0.05


class McTractor(App):
    """Base class for the main Kivy app."""

    amiga_state = StringProperty("???")
    amiga_speed = StringProperty("???")
    amiga_rate = StringProperty("???")


    STREAM_NAMES = ["rgb", "disparity", "left", "right"]

    def __init__(
        self,
        service_config: EventServiceConfig,
    ) -> None:
        super().__init__()

        self.counter: int = 0

        self.service_config = service_config

        self.async_tasks: list[asyncio.Task] = []

        # self.image_decoder = ImageDecoder()
        self.image_decoder = TurboJPEG()

        self.view_name = "rgb"

        self.touchCnt=0
        self.firstPos = [10,10]
        self.secondPos = [50,50]

        self.max_speed: float = 1.0
        self.max_angular_rate: float = 1.0

        self.errPrev = 0

        self.omega = 0
        self.linear = 0

    def build(self):
        def on_touch_down(window: Window, touch: MouseMotionEvent) -> bool:
            """Handles initial press with mouse click or touchscreen."""			
            if isinstance(touch, MouseMotionEvent) and int(
                os.environ.get("DISABLE_KIVY_OUSE_EVENTS", 0)
            ):		  
                return True
                
            for w in window.children[:]:
                if w.dispatch("on_touch_down", touch):
                    return True			 
                                      
            #print(self.root.current_tab.text)
                
            
            if self.root.ids['rgb'].collide_point(*touch.pos):		   
                # if the active tab is "Rgb" then process the touch wuthout this
                # there is a crossover with touch events on images of other tabs 
                # (probably due to the overalap of coords)
                if (self.root.ids["mainTab"].current_tab.text == "Crop"):	 
                    # The touch has occurred inside the widgets area. Do stuff!
                    sizeXim=float(self.root.ids['rgb'].size[0])
                    normsizeXim=float(self.root.ids['rgb'].norm_image_size[0])
                    sizeYim=float(self.root.ids['rgb'].size[1])
                    normsizeYim=float(self.root.ids['rgb'].norm_image_size[1])
                    x0=(sizeXim-normsizeXim)/2.0
                    y0=(sizeYim-normsizeYim)/2.0
                    minX=min(sizeXim,normsizeXim)
                    minY=min(sizeYim,normsizeYim)
                    posTmp=((float(touch.pos[0])-x0)/minX,(float(touch.pos[1])-y0)/minY)
                    if (self.touchCnt==0):
                        self.firstPos=posTmp
                        print(posTmp)
                        self.touchCnt=self.touchCnt+1
                    elif (self.touchCnt==1): 
                        self.secondPos=posTmp
                        print(posTmp)
                        self.touchCnt=self.touchCnt+1
                    else:
                        self.touchCnt=0
                pass					   
                
            # Add additional on_touch_down behavior here
            return False

        Window.bind(on_touch_down=on_touch_down)
        # end of bild() method
        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    def update_view(self, view_name: str):
        self.view_name = view_name

    async def app_func(self):
        async def run_wrapper() -> None:
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.async_tasks:
                task.cancel()

        config_list = proto_from_json_file(
            self.service_config, EventServiceConfigList()
        )

        oak0_client: EventClient | None = None
        canbus_client: EventClient | None = None


        for config in config_list.configs:
            if config.name == "oak0":
                oak0_client = EventClient(config)
            elif config.name == "canbus":
                canbus_client = EventClient(config)


        # Confirm that EventClients were created for all required services
        if None in [oak0_client,canbus_client]:
            raise RuntimeError(
                f"No {config} service config in {self.service_config}"
            )

        # Camera task
        self.tasks: list[asyncio.Task] = [
            asyncio.create_task(self.stream_camera(oak0_client, "rgb"))
            # for view_name in self.STREAM_NAMES
        ]

        self.tasks.append(asyncio.ensure_future(self.pose_generator(canbus_client)))

        return await asyncio.gather(run_wrapper(),*self.tasks)

    async def stream_camera(
        self,
        oak_client: EventClient,
        view_name: Literal["rgb", "disparity", "left", "right"] = "rgb",
    ) -> None:

        """Subscribes to the camera service and populates the tabbed panel with all 4 image streams."""
        while self.root is None:
            await asyncio.sleep(0.01)

        rate = oak_client.config.subscriptions[0].every_n

        async for event, payload in oak_client.subscribe(
            SubscribeRequest(
                uri=Uri(path=f"/{view_name}"), every_n=rate
            ),
            decode=False,
        ):
            if view_name == "rgb":  # self.view_name:
                message = payload_to_protobuf(event, payload)
                try:
                    img = self.image_decoder.decode(message.image_data)
                    self.mainTabImagingFunction(img)

                except Exception as e:
                    logger.exception(f"Error decoding image: {e}")
                    continue

    async def pose_generator(self, canbus_client: EventClient, period: float = 0.02):
        """The pose generator yields an AmigaRpdo1 (auto control command) for the canbus client to send on the bus
        at the specified period (recommended 50hz) based on the onscreen joystick position."""
        while self.root is None:
            await asyncio.sleep(0.01)

        twist = Twist2d()
        
        # joystick: VirtualJoystickWidget = self.root.ids["joystick"]

        rate = canbus_client.config.subscriptions[0].every_n

        async for event, payload in canbus_client.subscribe(
            SubscribeRequest(uri=Uri(path="/state"), every_n=rate),
            decode=False,
        ):
            message = payload_to_protobuf(event, payload)
            tpdo1 = AmigaTpdo1.from_proto(message.amiga_tpdo1)
            self.linear = self.root.ids.velocitySlider.value
            # twist.linear_velocity_x = self.max_speed * joystick.joystick_pose.y
            # twist.angular_velocity = self.max_angular_rate * -joystick.joystick_pose.x
            print(f"Angular: {self.omega} Linear: {self.linear}")
            twist.linear_velocity_x = self.linear
            twist.angular_velocity = self.omega
            self.amiga_state = tpdo1.state.name
            self.amiga_speed = "{:.4f}".format(twist.linear_velocity_x)
            self.amiga_rate = "{:.4f}".format(twist.angular_velocity)

            await canbus_client.request_reply("/twist", twist)
            await asyncio.sleep(period)

    def mainTabImagingFunction(self, rgb):
        print("WHAT",rgb.shape)
        rgb = cv2.flip(rgb,1) 
        print("WHO",rgb.shape)

        color=(0,0,255)
 
        start_point = (int(self.firstPos[0]*rgb.shape[1]), 
            int(self.firstPos[1]*rgb.shape[0]))
        end_point = (int(self.secondPos[0]*rgb.shape[1]), 
            int(self.secondPos[1]*rgb.shape[0]))	


        if (self.view_name == "rgb"):
            if (self.touchCnt == 2):
                    rgb = cv2.rectangle(rgb, start_point, end_point, color, 10)		   
            data = rgb.tobytes()
            texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), icolorfmt="bgr")
            texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
            self.root.ids[self.view_name].texture = texture

        if (self.view_name == "filter") or (self.view_name == "features") or (self.view_name == "control"):
            if (self.touchCnt == 2):                 
                x1 = start_point[0]
                y1 = start_point[1]
                x2 = end_point[0]
                y2 = end_point[1]
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                max_y = max(y1, y2)
                min_y = min(y1, y2)
                croppedRGB = rgb[min_y:max_y,min_x:max_x]
                hsv = cv2.cvtColor(croppedRGB, cv2.COLOR_RGB2HSV)
                low_hsv = np.array([self.root.ids.slider_LR.value, self.root.ids.slider_LG.value, self.root.ids.slider_LB.value])               
                high_hsv  = np.array([self.root.ids.slider_HR.value, self.root.ids.slider_HG.value, self.root.ids.slider_HB.value])
                mask2 = cv2.inRange(hsv, low_hsv, high_hsv)
                ## ftd = filtered image
                ftd = cv2.bitwise_and(croppedRGB, croppedRGB, mask=mask2)
                ## diffusion of image k = iterations
                k = 2
                ## diffused = filtered+diffused image
                diffused = anisotropicDiffusion(ftd,0.075 ,20, k)
                
                if (self.view_name == "filter"):
                    # show cropped and color filtered images
                    cropdata = croppedRGB.tobytes()
                    croptexture = Texture.create(size=(croppedRGB.shape[1],croppedRGB.shape[0]), icolorfmt="bgr")
                    croptexture.blit_buffer(cropdata, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
                
                    diffdata= diffused.tobytes()
                    difftexture = Texture.create(size=(diffused.shape[1],diffused.shape[0]), icolorfmt="bgr")							 
                    difftexture.blit_buffer(diffdata, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)

                    self.root.ids["cropImage"].texture = croptexture
                    self.root.ids["procImage"].texture = difftexture

                elif (self.view_name == "features") or (self.view_name == "control") or (self.view_name == "graphrgb"):                     
                    ## convertin the diffused image to a gray image and bw image
                    grayImg = cv2.cvtColor(diffused, cv2.COLOR_BGR2GRAY)
                    grayTreshold = self.root.ids.sliderGrayThreshold.value  
                    ret, bwImg = cv2.threshold(grayImg, grayTreshold, 255, cv2.THRESH_BINARY)
                    
                    ## finding blobs
                    contours, hierarchy = cv2.findContours(bwImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # print("contours #:",len(contours))
                    
                    lowRadius = self.root.ids.sliderRmin.value 
                    highRadius = self.root.ids.sliderRmax.value
                    #bwImgWithContoursRGB = cv2.drawContours(bwImgRGB, contours, -1, (0, 0, 255), 3)                       
                    bwImgWithBlobsRGB = cv2.cvtColor(bwImg, cv2.COLOR_GRAY2BGR)
                    imgHeight = bwImgWithBlobsRGB.shape[0]
                    imgWidth = bwImgWithBlobsRGB.shape[1]
                    lowRadius = self.root.ids.sliderRmin.value * max(imgHeight,imgWidth)  
                    highRadius = self.root.ids.sliderRmax.value * max(imgHeight,imgWidth)
                    cntCircle = 0
                    if (len(contours) > 0):
                        circleInfo = np.zeros([len(contours),3])                   
                        for i in range(len(contours)):
                            cnttmp = contours[i]
                            (x,y), radius = cv2.minEnclosingCircle(cnttmp)
                            if (radius > lowRadius) and (radius < highRadius):
                                cntCircle = cntCircle + 1
                                circleInfo[cntCircle-1,0] = int(x)
                                circleInfo[cntCircle-1,1] = int(y)
                                circleInfo[cntCircle-1,2] = int(radius)
                                center = (int(x),int(y))
                                radius = int(radius)
                                bwImgWithBlobsRGB = cv2.circle(bwImgWithBlobsRGB, center, radius,(0,0,255),3)
                    if (cntCircle > 0):
                        sumCircleRadius = 0.0
                        sumWeightCircle = 0.0
                        for i in range(cntCircle):
                            radius = circleInfo[i,2]
                            x = circleInfo[i,0]
                            sumCircleRadius = sumCircleRadius + radius * radius * 1.0
                            sumWeightCircle = sumWeightCircle + radius * radius * x * 1.0
                        M = (sumWeightCircle / sumCircleRadius)
                        intM = int(M)
                        bwImgWithBlobsRGB = cv2.line(bwImgWithBlobsRGB, (intM,0), (intM,imgHeight), (0, 0,255), 2)
                    else:
                        M = -1                        

                    if (self.view_name == "features"):
                        grayData = grayImg.tobytes()
                        grayTexture = Texture.create(size=(grayImg.shape[1],grayImg.shape[0]), icolorfmt="bgr")
                        grayTexture.blit_buffer(grayData, bufferfmt="ubyte", colorfmt="luminance", mipmap_generation=False)
                        
                        bwData = bwImgWithBlobsRGB.tobytes()
                        bwTexture = Texture.create(size=(bwImgWithBlobsRGB.shape[1],bwImgWithBlobsRGB.shape[0]), icolorfmt="bgr")
                        bwTexture.blit_buffer(bwData, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)

                        self.root.ids["grayImage"].texture = grayTexture
                        self.root.ids["bwImage"].texture = bwTexture
                    elif (self.view_name == "control") or (self.view_name == "graphrgb"):
                        # print("control")
                        intrefLine = int(imgWidth * self.root.ids.refLineSlider.value / 100.0)
                        
                        if (M > -1): #circles exist
                            self.errMes = intrefLine - M
                        else: #circles do not exist
                            self.errMes = 0

                        #computes new filtered error
                        alpha = self.root.ids.alphaSlider.value / 100.0                        
                        self.err = alpha*self.errPrev + (1-alpha)*self.errMes
                        self.errPrev = self.err 
                        errInt = int(self.err)

                        # Control u is turning radius and we compute is as u[rad/s]=Kp*err[px]
                        # Maximal possible err value is 1280px (image width), therefore is we assume that 
                        # the max turning rate is 10deg/s which is approx 0.17rad/s we get Kp = 0.0001328125. 
                        # Because of that we will compute u[rad/s]=Kp*err[px]/10000
                        # and we can use Kp range [0, 2] for intial testing

                        Kp = self.root.ids.kPSlider.value    
                        ctrlU = -Kp*self.err/10000
                        if (ctrlU >= 0.5):
                            ctrlU = 0.5
                        elif (ctrlU <= -0.5):
                            ctrlU = -0.5
                        else:
                            pass
                        
                        self.omega = ctrlU
                        # print(view_name,self.omega)

                        # _, encoded_image = cv2.imencode('.jpg', croppedRGB)

                        # timestamp = time.time()

                        # print(self.root.ids.recordState.state)
                        # if (self.root.ids.recordState.state == "down"):
                        #     print("down")
                        #     self.controllerStatus = 1
                        #     with open(self.bin_file_path, 'ab') as file:
                        #         data = {
                        #             'timestamp': timestamp,
                        #             'weighted_average': intM,
                        #             'error_meas': self.errMes,
                        #             'intref_line': intrefLine,
                        #             'control_effort': ctrlU,
                        #             'image': encoded_image,
                        #         }

                        #         pickle.dump(data, file)
                        # else:
                        #     self.controllerStatus = 0
                        #     print("normal")
                            # print("wrote to:",self.bin_file_path)
                        #written to csv (timestamp,self.omega)
                        #self.root.ids.maxTurningSlider.value = self.omega
                        
                        "Floating point {0:.2f}".format(345.7916732)
                        
                        self.root.ids.errorLabel.text="Error: {0:.3f}".format(self.err/10000)
                        self.root.ids.errorLabel.texture_update()
                        self.root.ids.controlLabel.text="Control: {0:.3f}".format(ctrlU)
                        self.root.ids.controlLabel.texture_update()

                        bwImgWithBlobsRGB = cv2.line(bwImgWithBlobsRGB, (intrefLine,0), (intrefLine,imgHeight), (0,255,255), 2)
                        bwImgWithBlobsRGB = cv2.line(bwImgWithBlobsRGB, (intrefLine-errInt,0), (intrefLine-errInt,imgHeight), (255,0,0), 5)
                        bwData = bwImgWithBlobsRGB.tobytes()
                        bwTexture = Texture.create(size=(bwImgWithBlobsRGB.shape[1],bwImgWithBlobsRGB.shape[0]), icolorfmt="bgr")
                        bwTexture.blit_buffer(bwData, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
                        self.root.ids["controlImage"].texture = bwTexture
                    else:
                        pass
                        # --------------------------------------------------------------       
                        # --- end of imagen processing og the cropped image in openCV ---
                        # -------------------------------------------------------------                    
                           



def find_config_by_name(
    service_configs: EventServiceConfigList, name: str
) -> EventServiceConfig | None:
    """Utility function to find a service config by name.

    Args:
        service_configs: List of service configs
        name: Name of the service to find
    """
    for config in service_configs.configs:
        if config.name == name:
            return config
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="template-app")

    # Add additional command line arguments here
    parser.add_argument("--service-config", type=Path, default="service_config.json")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(McTractor(args.service_config).app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
