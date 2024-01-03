# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import argparse
import asyncio
import io
import os
import numpy as np

from typing import List
from pathlib import Path
from cv2.ximgproc import anisotropicDiffusion

import grpc

import time
import pickle

#imports for canbus
from farm_ng.canbus import canbus_pb2
from farm_ng.canbus.canbus_client import CanbusClient
from farm_ng.canbus.packet import AmigaControlState
from farm_ng.canbus.packet import AmigaTpdo1
from farm_ng.canbus.packet import make_amiga_rpdo1_proto
from farm_ng.canbus.packet import parse_amiga_tpdo1_proto
from farm_ng.oak import oak_pb2

#imports for camera
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
from turbojpeg import TurboJPEG

os.environ["KIVY_NO_ARGS"] = "1"


from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.input.providers.mouse import MouseMotionEvent  # noqa: E402

from kivy.uix.tabbedpanel import TabbedPanel

# This is our app 
class CameraApp(App):
    def __init__(self, address: str, camera_port: int, canbus_port: int, stream_every_n: int) -> None:
        super().__init__()
        self.address = address
        self.camera_port = camera_port
        self.canbus_port = canbus_port
        self.stream_every_n = stream_every_n
        self.touchCnt=0
        self.firstPos = [10,10]
        self.secondPos = [50,50]
        self.controllerStatus = 0

        self.errMes=0
        self.err=0
        self.errPrev=0

        self.omega = 0

        # Received values (initializes communication ???)
        self.amiga_tpdo1: AmigaTpdo1 = AmigaTpdo1()

        # Parameters: max velocity and turning rate 
        self.max_speed: float = 1.0
        self.max_angular_rate: float = 1.0

        self.image_decoder = TurboJPEG() 

        # emty list of tasks 
        self.tasks: List[asyncio.Task] = []

        # define config sliders
        self.sliders = []
        self.featureSliders = []
        self.controlSliders = []
        #create CSV with name:__day__time__control_action
        #create CSV with name:__day__time__opencv_encode

        start_timestamp = time.time()
        directory = "/data/farm_ng/PickleData/"+ time.strftime('%d_%m', time.localtime(start_timestamp))
        print(directory)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)  # Create the directory
                print("Directory ", directory, " Created ")
            except OSError as error: 
                print("Directory Error:", error)
        
        start_time_str = time.strftime('%d_%H%M%S_%m%Y', time.localtime(start_timestamp))

        # Define the path for your .bin file, including the start time in the filename
        self.bin_file_path = os.path.join(directory, f'image_data_{start_time_str}.bin')
        print(self.bin_file_path)

        
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

    def save_config(self):
        # Save HSV configuration to txt file
        with open("cropConfig.txt", "w") as f:
            for pos in self.firstPos:
                f.write(str(pos))
                f.write("\n")
            for pos in self.secondPos:
                f.write(str(pos))
                f.write("\n")
        with open("HSVconfig.txt", "w") as f:
            for slider in self.sliders:
                f.write(str(slider.value))
                f.write("\n")
        with open("featureConfig.txt", "w") as f:
            for slider in self.featureSliders:
                f.write(str(slider.value))
                f.write("\n")
        with open("controlConfig.txt", "w") as f:
            for slider in self.controlSliders:
                f.write(str(slider.value))
                f.write("\n")

# -------------------- BEGIN ---------------------------------- 
# ---->  Function that processed mainTAB and Images       <----
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv         
    def mainTabImagingFunction(self, rgb):
        view_nameStr = self.root.ids["mainTab"].current_tab.text
        if view_nameStr == "Crop":
            view_name = "rgb"
        elif view_nameStr == "Filter":
            view_name = "cropped"     
        elif view_nameStr == "Features":
            view_name = "features"     
        elif view_nameStr == "Navigation":
            view_name = "control"
        elif view_nameStr == "Graphs":
            view_name = "graphrgb"
        else:
            view_name = "whatever" 
 
        self.activeTab = view_name

        rgb = cv2.flip(rgb,1) 
        color=(0,0,255)
 
        start_point = (int(self.firstPos[0]*rgb.shape[1]), 
            int(self.firstPos[1]*rgb.shape[0]))
        end_point = (int(self.secondPos[0]*rgb.shape[1]), 
            int(self.secondPos[1]*rgb.shape[0]))	
            # (0,0) bottom left corner
            # print(self.touchCnt,start_point,end_point) 
            # if the view is rgb image
        # print(view_name)
        if (view_name == "rgb"):
            if (self.touchCnt == 2):
                    rgb = cv2.rectangle(rgb, start_point, end_point, color, 10)		   
            data = rgb.tobytes()
            texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), icolorfmt="bgr")
            texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
            self.root.ids[view_name].texture = texture
            # the view is cropped image
        elif (view_name == "cropped") or (view_name == "control") or (view_name == "features") or (view_name == "graphrgb"):
            if (view_name == "graphrgb"):
                data = rgb.tobytes()
                texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), icolorfmt="bgr")
                texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
                self.root.ids[view_name].texture = texture
                
            if (self.touchCnt == 2):                 
                x1 = start_point[0]
                y1 = start_point[1]
                x2 = end_point[0]
                y2 = end_point[1]
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                max_y = max(y1, y2)
                min_y = min(y1, y2)
                #max_y = 1080 - max(y1, y2)
                #min_y = 1080 - min(y1, y2)
                #print('Start (x)?:',min_x,'start y?',max_y)
                #print('End (x)?:',max_x,'End (y)?',min_y)
                #crgb = rgb[max_y:min_y,min_x:max_x]
                croppedRGB = rgb[min_y:max_y,min_x:max_x]
                #croppedRGB = rgb[max_y:min_y,min_x:max_x]
                #croppedRGB = cv2.flip(croppedRGB,1)              
                # store (time,encoded_croppenRGB)
                        
                # ------------------------------------------------------------- 
                # ---->  image processing of the cropped image in openCV <----
                # -------------------------------------------------------------         
                #----------------------------# 
                #        Color filtering     #
                #----------------------------#
                ## crgb is the cropped image
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
                
                if (view_name == "cropped"):
                    # show cropped and color filtered images
                    cropdata = croppedRGB.tobytes()
                    croptexture = Texture.create(size=(croppedRGB.shape[1],croppedRGB.shape[0]), icolorfmt="bgr")
                    croptexture.blit_buffer(cropdata, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
                
                    diffdata= diffused.tobytes()
                    difftexture = Texture.create(size=(diffused.shape[1],diffused.shape[0]), icolorfmt="bgr")							 
                    difftexture.blit_buffer(diffdata, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)

                    self.root.ids["cropImage"].texture = croptexture
                    self.root.ids["procImage"].texture = difftexture
                
                elif (view_name == "features") or (view_name == "control") or (view_name == "graphrgb"):                     
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

                    if (view_name == "features"):
                        grayData = grayImg.tobytes()
                        grayTexture = Texture.create(size=(grayImg.shape[1],grayImg.shape[0]), icolorfmt="bgr")
                        grayTexture.blit_buffer(grayData, bufferfmt="ubyte", colorfmt="luminance", mipmap_generation=False)
                        
                        bwData = bwImgWithBlobsRGB.tobytes()
                        bwTexture = Texture.create(size=(bwImgWithBlobsRGB.shape[1],bwImgWithBlobsRGB.shape[0]), icolorfmt="bgr")
                        bwTexture.blit_buffer(bwData, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)

                        self.root.ids["grayImage"].texture = grayTexture
                        self.root.ids["bwImage"].texture = bwTexture
                    elif (view_name == "control") or (view_name == "graphrgb"):
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

                        _, encoded_image = cv2.imencode('.jpg', croppedRGB)

                        timestamp = time.time()

                        # print(self.root.ids.recordState.state)
                        if (self.root.ids.recordState.state == "down"):
                            print("down")
                            self.controllerStatus = 1
                            with open(self.bin_file_path, 'ab') as file:
                                data = {
                                    'timestamp': timestamp,
                                    'weighted_average': intM,
                                    'error_meas': self.errMes,
                                    'intref_line': intrefLine,
                                    'control_effort': ctrlU,
                                    'image': encoded_image,
                                }

                                pickle.dump(data, file)
                        else:
                            self.controllerStatus = 0
                            print("normal")
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
            #if the figure is not cropped
            else:
                # print("Not Cropped\n")
                if (view_name == "cropped") or (view_name == "features"): 
                        grayImg = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                        grayData = grayImg.tobytes()
                        grayTexture = Texture.create(size=(grayImg.shape[1],grayImg.shape[0]), icolorfmt="bgr")
                        grayTexture.blit_buffer(grayData, bufferfmt="ubyte", colorfmt="luminance", mipmap_generation=False)
                        
                        bwData = rgb.tobytes()
                        bwTexture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), icolorfmt="bgr")
                        bwTexture.blit_buffer(bwData, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)
                        
                        if (view_name == "cropped"):    
                            self.root.ids["cropImage"].texture = grayTexture
                            self.root.ids["procImage"].texture = bwTexture
                        elif (view_name == "features"):    
                            self.root.ids["grayImage"].texture = grayTexture
                            self.root.ids["bwImage"].texture = bwTexture
                        else: 
                            pass
                elif (view_name == "control"):
                        imgWidth = rgb.shape[1]
                        imgHeight = rgb.shape[0]
                        intrefLine = int(imgWidth * self.root.ids.refLineSlider.value / 100.0)
                        bwImgWithBlobsRGB = cv2.line(rgb, (intrefLine,0), (intrefLine,imgHeight), (0,255,255), 5)
                        bwData = bwImgWithBlobsRGB.tobytes()
                        bwTexture = Texture.create(size=(bwImgWithBlobsRGB.shape[1],bwImgWithBlobsRGB.shape[0]), icolorfmt="bgr")
                        bwTexture.blit_buffer(bwData, bufferfmt="ubyte", colorfmt="bgr", mipmap_generation=False)    
                        self.root.ids["controlImage"].texture = bwTexture
                else:
                    pass
        else:
            data = rgb.tobytes()
            texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), icolorfmt="bgr")
            texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")
            self.root.ids[view_name].texture = texture            
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         
# ---->  Function that processed mainTAB and Images       <----
# --------------------END--------------------------------------         


    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        # configure the camera client, adddress and port are variebles
        camera_config: ClientConfig = ClientConfig(
            address=self.address, port=self.camera_port)
        camera_client: OakCameraClient = OakCameraClient(camera_config)

        # configure the canbus client, adddress and port are variebles        
        canbus_config: ClientConfig = ClientConfig(
             address=self.address, port=self.canbus_port
        )
        canbus_client: CanbusClient =  CanbusClient(canbus_config)

        # Stream camera frames
        self.tasks.append(
            asyncio.ensure_future(self.stream_camera(camera_client))
        )

        # Canbus task(s)
        self.tasks.append(
            asyncio.ensure_future(self.stream_canbus(canbus_client))
        )
        self.tasks.append(
            asyncio.ensure_future(self.send_can_msgs(canbus_client))
        )

        return await asyncio.gather(run_wrapper(), *self.tasks)
    
    # Stream canbus task
    async def stream_canbus(self, client: CanbusClient) -> None:
        """This task:
        - listens to the canbus client's stream
        - filters for AmigaTpdo1 messages
        - extracts useful values from AmigaTpdo1 messages
        """
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None

        while True:
            # check the state of the service
            state = await client.get_state()

            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None

                print("Canbus service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            if (
                response_stream is None
                and state.value != service_pb2.ServiceState.UNAVAILABLE
            ):
                # get the streaming object
                response_stream = client.stream()

            try:
                # try/except so app doesn't crash on killed service
                response: canbus_pb2.StreamCanbusReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                # print(e)
                response_stream.cancel()
                response_stream = None
                continue

            for proto in response.messages.messages:
                amiga_tpdo1: Optional[AmigaTpdo1] = parse_amiga_tpdo1_proto(proto)
                if amiga_tpdo1:
                    # Store the value for possible other uses
                    self.amiga_tpdo1 = amiga_tpdo1

    # Send canbus message task
    async def send_can_msgs(self, client: CanbusClient) -> None:
        """This task ensures the canbus client sendCanbusMessage method has the pose_generator it will use to send
        messages on the CAN bus to control the Amiga robot."""
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None
        while True:
            # check the state of the service
            state = await client.get_state()

            # Wait for a running CAN bus service
            if state.value != service_pb2.ServiceState.RUNNING:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                # print("Waiting for running canbus service...")
                await asyncio.sleep(0.1)
                continue

            if response_stream is None:
                print("Start sending CAN messages")
                response_stream = client.stub.sendCanbusMessage(self.pose_generator())

            try:
                async for response in response_stream:
                    # Sit in this loop and wait until canbus service reports back it is not sending
                    assert response.success
            except Exception as e:
                # print(e)
                response_stream.cancel()
                response_stream = None
                continue

            await asyncio.sleep(0.1)

    async def pose_generator(self, period: float = 0.02):
        """The pose generator yields an AmigaRpdo1 (auto control command) for the canbus client to send on the bus
        at the specified period (recommended 50hz) based on the onscreen joystick position."""
        while self.root is None:
            await asyncio.sleep(0.01)

        #joystick: VirtualJoystickWidget = self.root.ids["joystick"]
        while True:
            msg: canbus_pb2.RawCanbusMessage = make_amiga_rpdo1_proto(
                state_req=AmigaControlState.STATE_AUTO_ACTIVE,
                cmd_speed=self.max_speed * self.root.ids.velocitySlider.value * self.controllerStatus,
                cmd_ang_rate = self.max_angular_rate * self.omega * -self.controllerStatus,
            )
            
            yield canbus_pb2.SendCanbusMessageRequest(message=msg)
            await asyncio.sleep(period)

    async def stream_camera(self, client: OakCameraClient) -> None:
        """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
        from the oak camera."""

        while self.root is None:
            await asyncio.sleep(0.01)


        # Load HSV configuration
        self.sliders = [self.root.ids.slider_LR, self.root.ids.slider_HR, self.root.ids.slider_LG, self.root.ids.slider_HG, self.root.ids.slider_LB, self.root.ids.slider_HB] 
        self.featureSliders = [self.root.ids.sliderGrayThreshold, self.root.ids.sliderRmin, self.root.ids.sliderRmax]
        self.controlSliders = [self.root.ids.refLineSlider, self.root.ids.alphaSlider, self.root.ids.kPSlider, self.root.ids.maxTurningSlider, self.root.ids.velocitySlider]
        
        #self.root.ids.refLineSlider, self.]
        iterateSliders = iter(self.sliders)
        iterateFeature = iter(self.featureSliders)
        iterateControl = iter(self.controlSliders)

        try:
            with open("cropConfig.txt", "r") as f:
                self.firstPos[0] = float(f.readline())
                self.firstPos[1] = float(f.readline())
                self.secondPos[0] = float(f.readline())
                self.secondPos[1] = float(f.readline())
                self.touchCnt = 2
            with open("HSVconfig.txt", "r") as f:
                for line in f:
                    # print(line)
                    next(iterateSliders).value = int(line) 
            with open("featureConfig.txt", "r") as f:
                for line in f:
                    next(iterateFeature).value = float(line)
            with open("controlConfig.txt", "r") as f:
                for line in f:
                    next(iterateControl).value = float(line)
        except FileNotFoundError:
            print("Configuration file not")

        response_stream = None

        while True:
            # check the state of the service
            state = await client.get_state()

            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                print("Camera service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            # Create the stream
            if response_stream is None:
                response_stream = client.stream_frames(every_n=self.stream_every_n)

            try:
                # try/except so app doesn't crash on killed service
                response: oak_pb2.StreamFramesReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                # print(e)
                response_stream.cancel()
                response_stream = None
                continue

            # get the sync frame
            # not the same frame that we're using
            frame: oak_pb2.OakSyncFrame = response.frame
            try:
                img = self.image_decoder.decode( getattr(frame, "rgb").image_data)
                rgb = img

                self.mainTabImagingFunction(rgb)        

            except Exception as e:
                # print(e)
                continue           


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--cameraport", type=int, required=True, help="The camera port.")
    parser.add_argument(
        "--canbusport",
        type=int,
        required=True,
        help="The grpc port where the canbus service is running.",
    )

    parser.add_argument(
        "--address", type=str, default="localhost", help="The camera address"
    )
    parser.add_argument(
        "--stream-every-n", type=int, default=1, help="Streaming frequency"
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            CameraApp(args.address, args.cameraport, args.canbusport, args.stream_every_n).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()