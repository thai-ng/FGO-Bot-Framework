# This is a bot framework for Fate Grand Order.
## Roadmap:
1. Detect Servants present in battle scene.
2. Select servant skills
3. Detect cards in card selection scene
4. Select cards
5. Detect suit skills
6. Select suit skills
7. Detect NP cards
8. Select NP cards
9. Build scripting framework for scripting battle actions
10. Build Android App that consumes this framework.
11. Explore outside of battle scripting

## TODOs:
### Build coordinate primitives:
1. Get skill rect coordinates
2. Get Start button coordinate
3. Get face cards coordinates
4. Detect NP cards & Get NP cards coordinate
5. Get suit button coordinate
5. Get suit skills coordinates
### Touch injection
1. Get touch coordinate for each element
2. Get basic touch injection working
3. Touch injection from screen space
### Android port
1. Get Android project started.
2. Get Android screenshot with accessibility service
3. Get Android touch injection with accessibility service
4. Build library for Android app consumption
5. Android app entry and UI.
### GPU CV
1. Explore OpenCV using GPU (CUDA only? or use phone GPU as well?)
2. Port library to GPU
3. Test library on phone