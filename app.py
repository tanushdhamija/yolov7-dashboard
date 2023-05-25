import streamlit as st
from main import Yolov7
import torch
import os


def main():

    st.title("Dashboard")
    inference_msg = st.empty()
    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio("Select input source", 
    ('RTSP/HTTPS', 'Webcam', 'Local video'))

    conf_thres = float(st.sidebar.text_input("Detection confidence", "0.50"))
    save_output_video = st.sidebar.radio("Save output video?",('Yes', 'No'))
    if save_output_video == 'Yes':
        save_img = True
    else:
        save_img = False
    

    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Local video":

        video = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)

        # save video temporarily to process it using cv2
        if video is not None:
            if not os.path.exists('./tempDir'):
                os.makedirs('./tempDir')
            with open(os.path.join(os.getcwd(), "tempDir", video.name), "wb") as file:
                file.write(video.getbuffer())
            
            video_filename = f'./tempDir/{video.name}'
        

        if st.sidebar.button("Run"):
            stframe = st.empty()

            st.subheader("Inference Stats")
            if1, if2 = st.columns(2)

            st.subheader("System Stats")
            ss1, ss2, ss3 = st.columns(3)

            # Updating Inference results
            with if1:
                st.markdown("**Frame Rate**")
                if1_text = st.markdown("0")
            
            with if2:
                st.markdown("**Detected objects in current frame**")
                if2_text = st.markdown("0")
            
            
            # Updating System stats
            with ss1:
                st.markdown("**Memory Usage**")
                ss1_text = st.markdown("0")

            with ss2:
                st.markdown("**CPU Usage**")
                ss2_text = st.markdown("0")

            with ss3:
                st.markdown("**GPU Memory Usage**")
                ss3_text = st.markdown("0")


            # Run
            local_run = Yolov7(source=video_filename, save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)

            local_run.detect()
            inference_msg.success("Inference Complete!")

            # delete the saved video
            if os.path.exists(video_filename):
                os.remove(video_filename)


    # -------------------------- WEBCAM ----------------------------------
    if input_source == "Webcam":
        
        if st.sidebar.button("Run"):

            stframe = st.empty()

            st.subheader("Inference Stats")
            if1, if2 = st.columns(2)

            st.subheader("System Stats")
            ss1, ss2, ss3 = st.columns(3)

            # Updating Inference results
            with if1:
                st.markdown("**Frame Rate**")
                if1_text = st.markdown("0")
            
            with if2:
                st.markdown("**Detected objects in current frame**")
                if2_text = st.markdown("0")
            
            # Updating System stats
            with ss1:
                st.markdown("**Memory Usage**")
                ss1_text = st.markdown("0")

            with ss2:
                st.markdown("**CPU Usage**")
                ss2_text = st.markdown("0")

            with ss3:
                st.markdown("**GPU Memory Usage**")
                ss3_text = st.markdown("0")

            # Run
            webcam_run = Yolov7(source='0', save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)
            
            webcam_run.detect()


    # -------------------------- RTSP/HTTPS ------------------------------
    if input_source == "RTSP/HTTPS":
        
        rtsp_input = st.sidebar.text_input("Video link")

        if st.sidebar.button("Run"):
    
            stframe = st.empty()

            st.subheader("Inference Stats")
            if1, if2 = st.columns(2)

            st.subheader("System Stats")
            ss1, ss2, ss3 = st.columns(3)

            # Updating Inference results
            with if1:
                st.markdown("**Frame Rate**")
                if1_text = st.markdown("0")
            
            with if2:
                st.markdown("**Detected objects in current frame**")
                if2_text = st.markdown("0")
            
            # Updating System stats
            with ss1:
                st.markdown("**Memory Usage**")
                ss1_text = st.markdown("0")

            with ss2:
                st.markdown("**CPU Usage**")
                ss2_text = st.markdown("0")

            with ss3:
                st.markdown("**GPU Memory Usage**")
                ss3_text = st.markdown("0")

            # Run
            stream_run = Yolov7(source=rtsp_input, save_img=save_img, conf_thres=conf_thres,
                            stframe=stframe, if1_text=if1_text, if2_text=if2_text,
                            ss1_text=ss1_text, ss2_text=ss2_text, ss3_text=ss3_text)
                            
            stream_run.detect()

    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass


