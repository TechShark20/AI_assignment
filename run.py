import cv2 
from  utils import mark_frame,make_df,Scale_exactmid #function to mark the middle coordinates 
from  utils import preprocess_dataframe
from utils import function_to_mark_the_needy

from detr import detect_balls
from ast import literal_eval as lit 
import pandas as pd 
def create_dataframe():
     df_log ={}
     df_log['frame_no']=[]
     df_log['quadrant']=[]
     df_log['color']=[]
     df_log['box']=[]
     return df_log
def video_writer(path :str ,df :pd.DataFrame,fps:int):
     
    

    
     if type(df["box"][0][0]) is str: 
             df['box']=df['box'].apply(lit)
             df['frame_no']=df["frame_no"].apply(lambda x:int(x))
     
     re_w =720
     re_h=640
     frm_cnt=0
     cap=cv2.VideoCapture(path)
     fps=int(cap.get(cv2.CAP_PROP_FPS))
     write_path ="output_file.mp4"
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     out = cv2.VideoWriter(write_path, fourcc, fps, (720,640))
    
     while True:
          ret,frame =cap.read()
          if not ret:
               print("video completed !!")
               break
          frm_cnt+=1
          print(frm_cnt)
          df_new=df[df["frame_no"]==frm_cnt]
         
          if(len(df_new)!=0):
              
              frame= function_to_mark_the_needy(frame,df_new)
          frame=cv2.resize(frame,(720,640))
          out.write(frame)
     cap.release()
     out.release()
     








          

     

def create_logs(path :str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    df_log=create_dataframe()
    re_w=720 #resized_w
    re_h=640 #resized_h
    print(fps)
    ret,frame =cap.read()
    hi,wi,_ =frame.shape #orignal hieght ,weight
    frame =cv2.resize(frame, (720,640))
     
    coordinates=mark_frame(frame)
    cap.release()
    if len(coordinates) != 2:
        raise Exception ("please select appropriate coordinates")
    mid_x_exact,mid_y_exact=Scale_exactmid(coordinates[0],coordinates[1],wi,hi,re_w,re_h)
    print(mid_x_exact,mid_y_exact)
    frame=None
    ret=None
    frm_cnt=0
    cap=cv2.VideoCapture(path)
    while True:
        
        ret,frame=cap.read()
        if not ret :
            print("video completed!")
            break
        frm_cnt+=1
        
        #frame=cv2.resize(frame,(720,640))
        boxes=detect_balls(frame)
        if len(boxes)!=0:
                 df_log=make_df(mid_x_exact,mid_y_exact,frm_cnt=frm_cnt,img=frame,df_log=df_log,boxes=boxes)
        #print(boxes)
        #frame=make_rectangles(frame,boxes)
        #frame =cv2.resize(frame,(720,640))
        #cv2.imshow("img",frame)
        #out.write(frame)
        #cv2.waitKey(0)

      
        print("the no. of frames processed are {}".format(frm_cnt))

    df_log = pd.DataFrame(df_log)
    df_log.to_csv("logs.csv",index=False)
    cap.release()
    return df_log,fps,total_frames
   































def main():
    video_file_path = 'AI Assignment video.mp4'
    df_log,fps,total_frames =create_logs(video_file_path)
    fps=30
    #df_log=pd.read_csv("log.csv")
    df_result,df_processing=preprocess_dataframe(fps,df_log)
    print("saving the entries with time stamps")
    df_result.to_csv("final_result.csv")
    #df_processing.to_csv("your_logs_orefined.csv")
    #function to create videos based on data_preprocessed
    video_writer(video_file_path,df_processing,fps)
    print("Execution completed!!")


    

    






    







if __name__ == "__main__":
    main()
