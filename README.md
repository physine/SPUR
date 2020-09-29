# SPUR scholarship reserch project
### This is my SPUR 2020 project.  It combines ARcore and TFlite in an Android app to track objects in the real world.
---

#### Usage

First the app needs to be built and installed from Android Studios.

Start the app.  Once started, it works best when the phone's camera is facing down at a slight angle at a surface and held moderately steady.  


Once a surface needs to be detected.  To do this point the camera at the surface to be detected.  ARcore detects surfaces by identifying unique features on the surface, 
so trying to detect a surface like a plane white table top will not work well.

When enough feature points have been grouped closely together a detected surface will begin to grow.

Once a surface is detected a tap on the screen will indicate the frame from the camera feed to process and use as input to the object recognition model.
The output of the model includes the label of the object detected and location of the detected object.  The location information is used to preform a hitTest at that location, if
successful an android robot is placed at the location for visual feedback.
This label is displayed to the screen along with the number of anchors placed.  An anchor is just a fixed location on the detected surface.  



Have a look at my poster!
[ClearyA-ComputerScience-2020.pdf](https://github.com/physine/SPUR/files/5299304/ClearyA-ComputerScience-2020.pdf)

Or watch my video presentation and demo on YouTube! https://youtu.be/rXudDBucquo

