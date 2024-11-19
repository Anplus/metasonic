# MetaSonic: Advancing Robot Localization with Directional Embedded Acoustic Signals

Indoor positioning in environments where GPS cannot be used is a fundamental technology for robotics navigation and human-robot interaction. However, existing vision-based localization systems cannot work in dark environments, and existing wireless or acoustic localization systems require specific transceivers, making them expensive and power-intensive — particularly challenging for micro-robots.
This paper proposes a new metasurface-assisted ultrasound positioning system. The key idea is to use a low-cost passive acoustic metasurface to transfer any speaker into a directional ultrasound source, with the acoustic spectrum varying based on direction. This allows any microrobot with a simple, low-cost microphone to capture such modified sound to identify the direction of the sound source. We develop a lightweight convolutional neural network-based localization algorithm that can be efficiently deployed on low-power microcontrollers. 

## Anchor Depoyment
The deployment of anchors is as follows. 12 anchor are deployed in the whole scene.
![Scenario](./img/scene.png)
