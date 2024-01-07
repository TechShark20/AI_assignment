
import cv2
import pandas as pd 
import numpy as np
from ast import literal_eval as lit
col_name=["color","quote","Hex_code","R","G","B"]
    # file from https://github.com/codebrainz/color-names
df_colors=pd.read_csv("colors.csv",header=None,names=col_name)
df_colors=df_colors.loc[(df_colors['color']=="raw_umber")|(df_colors['color']=="burnt_sienna")|(df_colors['color']=="msu_green")|(df_colors['color']=="dim_gray")]


coordinates=[]
def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append(x)
        coordinates.append(y)


def mark_frame(img):
    #"Clicking the centre of the frame :"
    # inp: an image of certain shape 
    # o/p : coordinate x and y (middle of the quadrants )
    #to store the middle 
    """
The idea is to manually fit the coordinates 
for identifying the quadrants 
"""
    
    global coordinates
    cv2.namedWindow("Mark_coodrinates")
    cv2.imshow("Mark_coordinates",img)
    cv2.setMouseCallback("Mark_coordinates", click_event)
    cv2.waitKey(0)
    cv2.circle(img, (coordinates[0], coordinates[1]), 2, (0, 255, 255))
    cv2.imshow("Mark_coordinates",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coordinates[0:2]

def make_rectangles(img,boxes):
    for i in boxes:
      img=cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 1)
      mid_x =(i[0]+i[2])//2
      mid_y = (i[1]+i[3])//2
      print(mark_color(img[mid_y,mid_x]))
    return img 


def mark_color (img_pixel):
    
    #convert B_G_R to R_G_B
    comp= np.array([img_pixel[2],img_pixel[1],img_pixel[0]])
    df_colors["Max_close"]=df_colors.apply(lambda x:np.linalg.norm(np.array([x["R"],x["G"],x["B"]])-comp),axis=1)
    closest_color = df_colors.loc[df_colors["Max_close"].idxmin()]
    return closest_color['color']

def Scale_exactmid(x,y,orig_w,orig_h,resized_w,resized_h):
    """
    This function is crucial for marking quadrant's
    We are using a resized window for mid point marking 
    inp:: the mid_point ,orig dimension ,resized dimension
    expected o/p:: mid_point according to orginal output

    """
    scaling_factor_x = orig_w/resized_w
    scaling_factor_y = orig_h /resized_h
    x_mid = int(x * scaling_factor_x)
    y_mid = int(y * scaling_factor_y)

    return x_mid,y_mid

def decide_quadrant (mid_x,mid_y,x,y):

    # we will decide quandrant accoding 
    # to the midpoint of the function
    # inp :: midpoints,ball_coord ,ops :: string 
    if mid_x>x and mid_y>y:
        return 4 #quadrant 4 acc. to video
    if mid_x>x and mid_y<y:
        return 1 #quadrant 1 acc. to video
    if mid_x<x and mid_y<y:
        return 3 #quadrant 3 acc. to video
    if mid_x<x and mid_y>y:
        return 2 #quadrant 2 acc. to video    

def make_df(mid_x,mid_y,frm_cnt,img,df_log,boxes):
    """
    i am unsure how to track multiple objects but approach is to log the
    tracking per frame then convert it into time 
    inp: boxes,midpoints_screen,frame_cnt,colour,df_log to store dictionary 

    return : list of dictionary 
    {
    frame_no : the curr fram
    quadrant: no.of quadrant
    color : color of the ball 
    box : the bounding box of the ball 
    }
    """
    
    for i in boxes :
       
        x_0,y_0,x_1,y_1 =i
        mid_obj_x=(x_0+x_1)//2
        mid_obj_y=(y_0+y_1)//2
        quadrant=decide_quadrant(mid_x,mid_y,mid_obj_x,mid_obj_y)
        color =  mark_color(img[mid_obj_y,mid_obj_x])
        df_log['frame_no'].append(frm_cnt)
        df_log['quadrant'].append(quadrant)
        df_log['color'].append(color) 
        df_log['box'].append(i)
        
    return df_log 

def preprocess_dataframe(fps : float  ,df:pd.DataFrame):
    """The dataframe collected from log will be proprocessed here
    we will first proper the dataset 
    1. convert string boxes to list :: ast.literal_eval
    2. make use of frame_no and quadrant :: as colors are not bieng classified properly
    3. create a new_df to log enteries 
    """
    df['box']=df['box'].apply(lit)
    df['quadrant']=df["quadrant"].apply(lambda x:int(x))
    df['frame_no']=df["frame_no"].apply(lambda x:int(x))
    df_quad1=df.loc[df['quadrant']==1]
    df_quad2=df.loc[df['quadrant']==2]
    df_quad3=df.loc[df['quadrant']==3]
    df_quad4=df.loc[df['quadrant']==4]
    df_quad1=cal_entry_exit(df_quad1,1)
    df_quad2=cal_entry_exit(df_quad2,2)
    df_quad3=cal_entry_exit(df_quad3,3)
    df_quad4=cal_entry_exit(df_quad4,4)
    df_quad1=filter_result(df_quad1,int(fps))
    df_quad2=filter_result(df_quad2,int(fps))
    df_quad3=filter_result(df_quad3,int(fps))
    df_quad4=filter_result(df_quad4,int(fps))
    df_final=pd.concat([df_quad1,df_quad2,df_quad3,df_quad4],axis=0)
    df_final["time_stamp"]=df_final["frame_no"].apply(lambda x:float(x/fps))
    df_final["time_stamp"]=df_final["time_stamp"].round(2)
    df_final=df_final.reset_index(drop=True)
    df_final = df_final.sort_values(by='frame_no', ascending=True)
    df_result=df_final.drop(["frame_no","box"],axis=1)
    df_processing =processor_for_vid(df_final,int(fps))
    return df_result,df_processing
    









def marker(df,i,x :str,mark_entry_exit :dict):
  
  """utility for making entries in cal_entry function"""
  mark_entry_exit["quadrant"].append(df.loc[i,"quadrant"])
  mark_entry_exit["frame_no"].append(df.loc[i,"frame_no"])
  mark_entry_exit["color"].append(df.loc[i,"color"])
  mark_entry_exit["type"].append(x)
  mark_entry_exit["box"].append(df.loc[i,"box"])     
  return mark_entry_exit                                       # function to fill the inputs of dictionary
   




def cal_entry_exit(df :pd.DataFrame,quadrant : int):
  
  """
  Marking entry exit based on color chnge 
  """
  mark_entry_exit={}
  mark_entry_exit["quadrant"]=[]
  mark_entry_exit["frame_no"]=[]
  mark_entry_exit["color"]=[]
  mark_entry_exit["type"]=[]
  mark_entry_exit["box"]=[]
  df =df.reset_index(drop=True)
  marker(df,0,"entry",mark_entry_exit=mark_entry_exit)
  #store the 0th index frame 
  for i in range(1,len(df)):
    df_prev=df.loc[i-1,['color']]
    df_curr=df.loc[i,['color']]
    if df_prev['color'] !=df_curr['color']:
      #compare if color changes than entry /exit 
      #exit of previous one and entry of new one 
      marker(df,i-1,"exit",mark_entry_exit)
      marker(df,i,"entry",mark_entry_exit)

  
  return pd.DataFrame(mark_entry_exit)


def filter_result(li :pd.DataFrame,fps : int ):
   li['frame_diff'] = li['frame_no'].diff()
   rows_to_keep = li[(li['frame_diff'] > 4*fps) | (li['frame_diff'].isnull())]
   df_filtered = rows_to_keep
   # Drop the temporary 'frame_diff' column
   df_filtered = df_filtered.drop(columns=['frame_diff'])
   # Reset index if needed
   df_filtered = df_filtered.reset_index(drop=True)
   return df_filtered
def processor_for_vid(df :pd.DataFrame,fps:int):
   """with the use of copy function we can efficiently make processing df 
      The idea is to hold the time_stamp ::-> for 1sec so for that one sec we will hold it 

   
   """
   df=df.drop(["quadrant","color"],axis=1)
   dfs_to_concat = [df]

   for i in range(0,len(df)):
      frm=df.loc[i].copy()
      flag = 1 if str(frm["type"]) =="entry" else -1
    
      for k in range(0,fps):
         frm["frame_no"]+=flag
         dfs_to_concat.append(pd.DataFrame(frm).transpose())
   return pd.concat(dfs_to_concat,ignore_index=True)


def function_to_mark_the_needy(image,df :pd.DataFrame):
   """
   This function will help me to preprocess the final result 
   by iter
   """
   for i,r in df.iterrows():
      box= r["box"]
      type=r["type"]
      time_stamp=r["time_stamp"]
      time_str="time::"+str(time_stamp)
      x_0,y_0,x_1,y_1 =box
      cv2.rectangle(image, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)
      cv2.putText(image, time_str, (x_0, y_0 -30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(image, type, (x_0-30, y_0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      return image

   



         


      










    



















