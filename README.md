YOLOv7 based dashboard to detect and view common objects through various sources such as RTSP, HTTPS, local video, webcam using streamlit.

## Usage
1. Clone the repo
2. Install all the dependencies in requirements.txt (in a python virtual environment (recommended))
3. Download the [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) weights and put them in the root folder
4. Run:
``` shell
streamlit run app.py
```
5. Access the web app on "localhost:8501"

## References
1. [Official YOLOv7 repo](https://github.com/WongKinYiu/yolov7)
2. [Streamlit dashboard](https://github.com/SahilChachra/Video-Analytics-Dashboard)
