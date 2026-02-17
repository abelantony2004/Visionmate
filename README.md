# INTRODUCTION

Navigating the environment independently remains one of the most significant challenges for individuals with visual impairments, especially in unfamiliar or obstacle-rich areas. Without the sense of sight, it is extremely difficult for the visually impaired to understand their surroundings and adapting with other senses has its own limitations as well. This project proposes the development of an AI-powered visual aid system that assists visually impaired users in detecting obstacles and navigating safely through indoor and outdoor environments while maintaining cost-effective and practical for use.

# BACKGROUND OF INVENTION

Background of the Invention

Navigating the environment independently is significantly challenging for individuals with visual impairments. While many can adapt to familiar surroundings with other senses, independently travelling and getting by in all environments is difficult due to a lack of public infrastructure accounting for the visual impaired and the need for a guide or assistance with navigation. 

Traditional physical assistive tools like white canes provide tactile feedback of objects surrounding a person however they cannot be used identify objects at a greater distance and the user may not be able to identify dangers. Blind walkways are also not widespread and makes pedestrian safety a critical drawback when dependent on traditional tools.

To address this challenge, the present invention provides a low-cost, practical, and AI-powered method for detecting obstacles and providing navigation for avoiding them. The invention utilizes cameras and ultrasonic sensors to detect and identify obstacles in the user’s path and prevent a risk of collision by communicating its presence to the user in the form of audio feedback. Obstacle detection helps the device to distinguish between harmful and harmless obstacles that needs to be warned about. The user can also use the device for audio guidance to their destination with GPS tracking, which simultaneously detects obstacles on the route. An OCR system is also used for identifying important signs and text in an environment.

# OBJECT OF THE INVENTION
The project aims to assist visually impaired individuals by enhancing their ability to navigate both indoor and outdoor environments independently. It addresses challenges such as obstacle detection, object and text recognition, pathfinding, delivering real-time feedback through audio cues to ensure user safety and situational awareness.

# BRIEF DESCRIPTION OF THE INVENTION
Brief Description of the Invention

The present invention is a low-cost and AI powered system that can be used by the visually impaired for obstacle detection and avoidance. The invention utilizes a camera and ultrasound sensor integrated with a Raspberry Pi computer to analyze the user’s environment for obstacles that needs to avoided by the user. The system recognizes walkways, footpaths, roadsides, rooms, and other pedestrian and indoor environments where it can find a route for the user to walk and detects obstacles on. 

When a user is walking on a footpath and an obstacle is detected by the sensor, the camera utilizes YOLO v8 to identify the obstacle. If the obstacle must be avoided to prevent injury to the user, audio feedback is generated using a Text-to-Speech service to warn the user of the obstacle. The system meanwhile uses SLAM algorithm to create a virtual environment replicating the user’s surroundings and locates the user in it. It then uses A* algorithm to find the shortest path avoiding the obstacle and the directions for these are also given to the user in the form of Text-to-Speech generated audio.

The camera is also utilized by the OCR system to read out important textual information for the user. The YOLO v8 model can identify objects like road signs, boards, caution signs, etc. that the user must be made aware of. Important text on them must also be accessible to the visually impaired. So the OCR system, scans text on detected objects, converts it to digital text and processes them before using the Text-to-Speech system too generate audio reading out the text.

This invention provides effective, fast, and smart assistance to the visually impaired for navigation in environments independently while ensuring their safety and providing them with better environmental awareness compared to traditional aid tools.

# CLAIMS 
1. A system for detecting obstacles and providing navigation to avoid them, comprising:
 a) a camera to capture video feed of user’s path;
 b) ultrasonic sensors to detect obstacle distance and for mapping virtual environment;
 c) GPS receiver to track user’s location and provide directions to destination and find route;
 d) speakers to output the audio feedback to user.
2. The system of claim 1, wherein the camera’s video is used to detect obstacles using yolo v8 and identify obstacles to be avoided. It is also used by SLAM to create mappings of outdoor environments
3. The system of claim 1, wherein ultrasonic sensors detect the obstacles distance to determine when danger will be encountered and provides input to SLAM where its ultrasonic feedback can be used to map an indoor environment using echolocation.
4. The system of claim 1, wherein the inputs are processed to identify obstacles in a mapped environment and A* algorithm is used to find a route avoiding the obstacle
5. The system of claim 1, wherein the route to avoid obstacle is converted to textual directions that is converted to audio using TTS and output to user.
6. The system of claim 1, further comprising a power supply connected to the system to enable portable and field-level operation.


# ABSTRACT

Navigating new environments poses significant challenges for visually impaired individuals,
often leading to safety concerns and reduced independence. Traditional aids like white canes and guide dogs, while effective, have limitations in providing deeper environmental information and real-time obstacle alerts.

This project addresses these challenges by proposing an AI-powered visual aid system designed to enhance independent navigation. The system integrates computer vision, deep learning, and sensor technologies to offer real-time obstacle detection, pathfinding assistance, and critical environmental information, thereby improving safety and fostering greater autonomy. Our work focuses on developing a robust and intuitive system that captures real-time video data through a wearable device with a camera module. This visual input is then processed using deep learning models for accurate object detection and classification, including obstacles, potholes, road signs, and walk lanes. Concurrently, ultrasonic sensors provide precise distance measurements for immediate obstacle proximity alerts, which are communicated to the user through audio feedback. The system also incorporates OCR for identifying signs and text, and text-to-speech (TTS) for audible communication of object types, locations, and navigation instructions. The core essence of this project lies in combining various environmental data through sensors and calculating safe and optimal paths for the user to take. The applications of this system are far-reaching, enabling visually impaired individuals to navigate both indoor and outdoor environments with increased confidence and safety. It can facilitate independent access to public spaces, enhance mobility in unfamiliar territories, and provide crucial information for daily tasks.
