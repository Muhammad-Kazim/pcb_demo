import matplotlib.pyplot as plt
import streamlit as st
import torch
from utils import *


st.set_page_config(
    page_title="PCB Demo",
)
st.title('PCB Analysis Demo')

STYLE = """
<style>
    img {
    max-width: 100%;
    }
</style>
"""

primaryColor = "#000000"

s = f"""
<style>
    div.stButton > button:first-child {{ border: 5px solid {primaryColor}; background-color: #D0F0C0; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)


def main():

    # st.sidebar.title("Choose the App Mode")
    app_mode = st.sidebar.radio(
        "Choose the app mode",
        ["Project Overview", "Classification", "FAQ"])

    if app_mode == "Project Overview":
        img_demo = Image.open('./images/demo_overview.png')
        # header1=st.header("Importance")
        placeholder1 = st.image(
            img_demo,
            width=700,
            caption="Classification: Specifically designed checks for anomaly detection."
        )

        img_flowchart = Image.open('./images/system_flowchart.png')
        header_flowchart = st.header("Project Overview")
        placeholder2 = st.image(
            img_flowchart,
            width=750,
            caption="System Design: Cascaded checks based on conventional computer vision working"
                    " in tandem with state of the art deep learning anomaly detection techniques" 
                    " tailored for PCBs."
        )
        #header3 = st.header("Labeling Demo")

        #video_file = open('./video.mp4', 'rb')
        #video_bytes = video_file.read()
        #placeholder3=st.video(video_bytes)

    elif app_mode == "Classification":

        st.subheader('Welcome to Classification Module')
        classification()

    elif app_mode == "FAQ":

        st.subheader("Frequently Asked Questions")
        faq()


def faq():
    e1 = st.expander("Why are the images so bad?")
    e2 = st.expander("Can you use a better camera?")
    e3 = st.expander("Can you describe the procedure?")
    e4 = st.expander("Why are you using AI? Isn't there a cheaper option?")
    e5 = st.expander("Can we enhance the image, that the worker can do a better/easier job?")
    e6 = st.expander("How much do we save?")


def classification():

    pred_obj = pred_pcb()
    st.sidebar.markdown("# Browse")
    data_type = st.sidebar.radio(
        "Do you want to upload an image or choose from database?",
        ["Database", "Upload"])

    if data_type == "Database":
        defect_notdefect_DB(pred_obj)

    if data_type == "Upload":
        defect_notdefect_O(pred_obj)


def defect_notdefect_DB(pred_obj):

    folder_path = "./dataset"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list = filenames1 + filenames
    selected_filename = st.sidebar.selectbox('Select a file', list)

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #000000;
        border-color: #ffffff;
    }
    </style>""", unsafe_allow_html=True)

    if selected_filename != 'None':

        file_path = os.path.join(folder_path, selected_filename)
        pil_img = Image.open(file_path)
        img_nd = preprocess(pil_img)

        st.image(img_nd, caption=f"{selected_filename}", width=400)
        # st.text("")
        _, col1, _, _, _ = st.columns([1, 1, 1, 1, 1])
        classify = col1.button('Classify', key=101)

        if classify:
            size_check, (corner_check, border_check), critical_region_check, arms_check, total_check = pred_obj.pred(file_path)

            # pil_img = Image.open("output.jpg")
            # img_nd = preprocess(pil_img)
            # st.image(img_nd, caption=f"{selected_filename}_output", width=1000)

            border = ''
            if border_check == corner_check and border_check == 'Pass':
                border = 'Pass'
            else:
                border = 'Fail'

            st.markdown(f"Classification: **{total_check}**")
            st.text(f"Size Check: {size_check}")
            st.text(f"Border Check: {border}")
            st.text(f"Critical Region Check: {critical_region_check}")
            st.text(f"Arms Check: {arms_check}")


def defect_notdefect_O(pred_obj):

    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "jpg"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)

    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }
        </style>""", unsafe_allow_html=True)

    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "jpg"]))
        return

    pil_img = Image.open(file)
    pil_img = pil_img.resize([2000, 1655])
    pil_img.save("upload.jpg")
    img_nd = preprocess(pil_img)
    st.image(img_nd, caption="Full Image", width=400)
    # st.text("")

    _, col1, _, _, _ = st.columns([1, 1, 1, 1, 1])
    classify = col1.button('Classify', key=101)

    if classify:
        size_check, (corner_check, border_check), critical_region_check, arms_check, total_check = pred_obj.pred(
            'upload.jpg')

        if border_check == corner_check and border_check == 'Pass':
            border = 'Pass'
        else:
            border = 'Fail'

        st.markdown(f"Classification: **{total_check}**")
        st.text(f"Size Check: {size_check}")
        st.text(f"Border Check: {border}")
        st.text(f"Critical Region Check: {critical_region_check}")
        st.text(f"Arms Check: {arms_check}")


def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


def preprocess(pil_img):
    img_nd = np.array(pil_img)

    if len(img_nd.shape)<3:
        img_nd=cv2.cvtColor(img_nd, cv2.COLOR_GRAY2RGB)
    # img_nd=pil_img
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    return img_nd


def HWCtoCHM(img_nd):
    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans


class pred_pcb:

    def __init__(self):

        current_directory = os.getcwd()
        self.arms_params = {
            "model": "padim",
            "config_path": os.path.join(current_directory, 'model/PCB_3.yaml'),
            "model_path": f"{current_directory}//checkpoint/pcb_3.ckpt",
            "threshold": 0.55
        }

        self.critical_params = {
            "model": "padim",
            "config_path": os.path.join(current_directory, 'model/PCB_1_Adjusted_275.yaml'),
            "model_path": f"{current_directory}//checkpoint/pcb_1_adjusted_275.ckpt",
            "threshold": 0.5
        }

        self.critical_region_obj = CriticalRegionDefects(self.critical_params["config_path"],
                                                         self.critical_params["model_path"])
        self.arm_defects_obj = ArmDefects(self.arms_params["config_path"], self.arms_params["model_path"])

    def pred(self, img_path):

        title_font_size = 16
        xlabel_font_size = 16

        pcb_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        start_time = time.time()
        processed_img, (height, width) = processing(img_path, manual_rotation=False, resize_fx=0.45, resize_fy=0.45,
                                                    padding=100)
        display_defect_img, defects_intensity, corner_dist = border_analysis(processed_image=processed_img,
                                                                             show_defects=True, defect_threshold=5,
                                                                             show_details=False, show_corners=True)

        cv2.imwrite('./temp_critical.jpg',
                    cv2.resize(processed_img[100:int(100 + height), 100:int(100 + width)], (275, 275)))
        cv2.imwrite('./temp_arms.jpg', processed_img[100:int(100 + height), 100:int(100 + width)])

        vis_img_rect, is_defect_in_critical, _ = self.critical_region_obj.infer('./temp_critical.jpg',
                                                                                self.critical_params["threshold"])
        image_for_arms, is_defect_on_arms = self.arm_defects_obj.infer('./temp_arms.jpg', self.arms_params["threshold"])

        size_check, (corner_check, border_check), critical_region_check, arms_check, total_check = checks(height, width,
                                                                                                          corner_dist,
                                                                                                          defects_intensity,
                                                                                                          is_defect_in_critical,
                                                                                                          is_defect_on_arms)

        end_time = time.time()
        total_time = end_time - start_time

        # fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        # axs[0].imshow(cv2.cvtColor(pcb_img, cv2.COLOR_BGR2RGB))
        # axs[0].set_title('Original Image', fontsize=title_font_size)

        plt.figure()
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.xlabel("Processed Image", fontsize=xlabel_font_size)
        plt.title(f"Size Check: {size_check}", fontsize=title_font_size)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig("col1.jpg")

        plt.figure()
        plt.imshow(cv2.cvtColor(display_defect_img, cv2.COLOR_BGR2RGB))
        plt.xlabel("Border Analysis", fontsize=xlabel_font_size)
        plt.title(f'Border Check: {"Pass" if border_check == corner_check == "Pass" else "Fail"}',
                         fontsize=title_font_size)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig("col2.jpg")

        plt.figure()
        plt.imshow(vis_img_rect)
        plt.xlabel("Critical Region Analysis", fontsize=xlabel_font_size)
        plt.title(f"Critical Region Check: {critical_region_check}", fontsize=title_font_size)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig("col3.jpg")

        plt.figure()
        plt.imshow(image_for_arms)
        plt.xlabel("Arms Analysis", fontsize=xlabel_font_size)
        plt.title(f"Arms Check: {arms_check}", fontsize=title_font_size)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig("col4.jpg")

        # fig.suptitle(f'Classification: {total_check}', fontsize=20)
        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        # plt.show()
        # plt.savefig("output.jpg")
        return size_check, (corner_check, border_check), critical_region_check, arms_check, total_check

if __name__ == "__main__":

    main()
