"""CSC 8830 Assignment 7, Task 1: calibrated stereo object measurement.

Run: streamlit run app.py
Use the Calibration tab first. Without calibration, estimates are explicitly
marked exploratory and do not satisfy the calibrated-stereo requirement.
"""
from pathlib import Path
import tempfile
import os
from room_video import frame_at, video_info, scale_camera_matrix, reconstruct_matched_points, point_pair_distance
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from stereo_geometry import decode_image, calibrate_single_camera, estimate_pose, triangulate_points, measure_shape, rotate_calibration_90

st.set_page_config(page_title='CSC 8830 · Stereo Measurement', layout='wide')
st.title('Module 7 · Calibrated Stereo Object Measurement')
st.caption('Task 1: calibrated stereo objects · Task 3: video-based room reconstruction')
BASE = Path(__file__).parent

SAMPLES = {
  'Journal / rectangle': ('journal_view1.jpeg', 'journal_view2.jpeg', 'Rectangle'),
  'Blue bowl / circle': ('lid_view1.jpeg', 'lid_view2.jpeg', 'Circle'),
  'Pink pentagon / polygon': ('pentagon_view1.jpeg', 'pentagon_view2.jpeg', 'Polygon'),
}


def as_pil(im):
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def marked(im, points):
    image = as_pil(im)
    d = ImageDraw.Draw(image)
    for i, (x,y) in enumerate(points):
        d.ellipse((x-8,y-8,x+8,y+8),fill='#ff4a52',outline='white',width=3)
        d.text((x+11,y-12),str(i+1),fill='white',stroke_width=2,stroke_fill='black')
    return image


def receive_image(upload, sample):
    return decode_image(upload.getvalue()) if upload is not None else cv2.imread(str(BASE/'sample_images'/sample))

cal_tab, measure_tab, room_tab, deriv_tab = st.tabs(['1 · Camera calibration', '2 · Object measurement', '3 · Room video analysis', '4 · Uncalibrated derivation (plan)'])

with cal_tab:
    st.subheader('Calibrate iPhone intrinsics')
    st.write('Capture **15–20 photos of a printed checkerboard** at different angles and positions with the SAME 1× lens and image resolution used for the object pairs. A checkerboard is a calibration target, not 20 objects.')
    a,b,c=st.columns(3)
    with a: nx=st.number_input('Inner corners across',min_value=3,max_value=20,value=9)
    with b: ny=st.number_input('Inner corners down',min_value=3,max_value=20,value=6)
    with c: square_cm=st.number_input('Printed square side (cm)',min_value=0.1,value=2.0,step=0.1)
    uploads=st.file_uploader('Checkerboard images (8 minimum, 15–20 recommended)',type=['jpg','jpeg','png'],accept_multiple_files=True)
    if st.button('Load and calibrate using the 20 included checkerboard photographs'):
        try:
            files=sorted((BASE/'calibration_images').glob('*.jpg'))
            K,dist,rms,n,size=calibrate_single_camera([cv2.imread(str(f)) for f in files],(int(nx),int(ny)),float(square_cm))
            st.session_state['camera_cal']=(K,dist,size)
            st.success(f'Actual calibration from {n} included photographs. Reprojection RMS = {rms:.3f} px. Verify the 2 cm printed squares and camera/video lens consistency.')
            st.dataframe(K)
        except Exception as e: st.error(str(e))

    if st.button('Run checkerboard calibration',type='primary'):
        try:
            imgs=[decode_image(u.getvalue()) for u in uploads]
            K,dist,rms,n,size=calibrate_single_camera(imgs,(int(nx),int(ny)),float(square_cm))
            st.session_state['camera_cal']=(K,dist,size)
            st.success(f'Calibrated using {n} images; reprojection RMS = {rms:.3f} pixels. Same camera/lens/resolution required.')
            st.dataframe(K)
        except Exception as e: st.error(str(e))
    if 'camera_cal' not in st.session_state:
        st.info('No checkerboard calibration loaded yet. Exploratory mode uses assumed intrinsics and is not a validated stereo result.')
    else:
        st.caption('Checkerboard intrinsics are loaded for this app session. Confirm square size, camera lens, and measurement accuracy independently.')

with measure_tab:
    example=st.selectbox('Choose your own captured object',list(SAMPLES))
    left_file,right_file=st.columns(2)
    with left_file: upl=st.file_uploader('Or replace view 1',type=['jpg','jpeg','png'],key='left_upload')
    with right_file: upr=st.file_uploader('Or replace view 2',type=['jpg','jpeg','png'],key='right_upload')
    left_name,right_name,shape_default=SAMPLES[example]
    L=receive_image(upl,left_name); R=receive_image(upr,right_name)
    if L is None or R is None: st.stop()
    if L.shape[:2]!=R.shape[:2]:
        st.error('Image dimensions differ. Re-export at the same resolution and camera settings.');st.stop()
    st.caption(f'Original sample files: View 1 = {left_name} · View 2 = {right_name}. These are separate captured images.')
    with st.expander('Compare original left and right photographs', expanded=False):
        preview_left, preview_right = st.columns(2)
        preview_left.image(as_pil(L), caption=f'View 1 · {left_name}', use_container_width=True)
        preview_right.image(as_pil(R), caption=f'View 2 · {right_name}', use_container_width=True)
    shape=st.selectbox('Object shape',['Rectangle','Circle','Polygon'],index=['Rectangle','Circle','Polygon'].index(shape_default))
    baseline=st.number_input('Camera displacement baseline (cm)', min_value=0.01, value=15.24,step=0.5,
           help='6 inches = 15.24 cm if this was the lens-to-lens displacement between the two shots. Confirm separately for each pair; camera-to-object distance (about 22 inches) is NOT the baseline.', key=f'baseline-{example}')
    st.caption('Reference notes: blue bowl diameter ≈ 4 in (10.16 cm); camera-to-object distance ≈ 22 in (55.88 cm); displacement ≈ 6 in (15.24 cm), as measured with iPhone Measure. Verify these values independently. Do not use the object size to tune the stereo result.')
    calibrated='camera_cal' in st.session_state
    orientation_setting='same resolution'
    if calibrated:
        K,dist,cal_size=st.session_state['camera_cal']
        photo_size=(L.shape[1],L.shape[0])
        if cal_size == photo_size:
            st.success('Loaded actual checkerboard intrinsics. Camera motion and baseline still require validation.')
        elif cal_size == photo_size[::-1]:
            st.info(f'Checkerboard calibration {cal_size} is portrait; object images {photo_size} are landscape. A 90° orientation transform is available.')
            rotation_choice=st.selectbox('How the checkerboard image is rotated to match the object image',
                ['Clockwise 90°', 'Counterclockwise 90°'],
                help='This refers to rotating the checkerboard IMAGE in pixel coordinates, not the direction you moved your phone. Verify orientation/lens with the original images.')
            orientation_setting=rotation_choice
            K,dist,rotated_size=rotate_calibration_90(K,dist,cal_size,
                'clockwise' if rotation_choice=='Clockwise 90°' else 'counterclockwise')
            assert rotated_size==photo_size
            st.success('Orientation-adjusted camera intrinsics loaded. Still validate results against independent reference measurements.')
        else:
            st.error(f'Calibration resolution {cal_size} does not match object photo resolution {photo_size}, even after a 90° rotation. Recalibrate with matching camera settings.');st.stop()
    else:
        st.warning('EXPLORATORY MODE — NO ACTUAL CAMERA CALIBRATION. Measurements are not suitable as validated assignment results.')
        approximate_f=st.number_input('Hypothetical focal length in pixels (exploration only)',min_value=100.,value=1250.,step=50.)
        K=np.array([[approximate_f,0,L.shape[1]/2],[0,approximate_f,L.shape[0]/2],[0,0,1]],dtype=float)
        dist=np.zeros((1,5),np.float64)
    pair_signature=(example, (upl.name, len(upl.getvalue())) if upl is not None else left_name, (upr.name, len(upr.getvalue())) if upr is not None else right_name, bool(calibrated), float(K[0,0]), orientation_setting)
    if st.session_state.get('pose_for') != pair_signature:
        st.session_state.pop('pose',None)
    if st.button('Estimate motion from feature matches'):
        try:
            rotation,direction,stats=estimate_pose(L,R,K,dist)
            st.session_state['pose']=(rotation,direction)
            st.session_state['pose_for']=pair_signature
            st.success(f"Camera pose estimated: {stats['pose_inliers']} inliers / {stats['matches']} matches. Translation scale remains estimated.")
            if abs(float(direction[2])) > 0.65:
                st.warning('Estimated camera movement points mostly forward/backward instead of sideways. This pair may not be suitable for reliable stereo measurements. Recheck or recapture with a controlled lateral motion.')
        except Exception as e:
            st.session_state.pop('pose',None);st.error(str(e))
    st.caption('Feature-based pose may be unstable for flat scenes. Verify movement and dimensions; do not treat a successful fit as proof of accuracy.')
    current=pair_signature
    if st.session_state.get('selected_example')!=current:
        st.session_state['selected_example']=current
        st.session_state['pts_l']=[];st.session_state['pts_r']=[]
    st.markdown('#### Select corresponding points in BOTH photographs')
    st.write('Rectangle: **top-left, top-right, bottom-right, bottom-left**. Circle: **two opposite points across the diameter**. Pentagon: **five corners clockwise**. Repeat the same order on both views.')
    st.caption('Zoom changes only the display: selected coordinates are mapped back to the ORIGINAL full-resolution image for 3D calculations.')
    zoom=st.checkbox('Zoom in for more precise corner selection', value=('pentagon' in example.lower()))
    side=st.radio('Place the next point on', ['View 1','View 2'],horizontal=True)
    pts_key='pts_l' if side=='View 1' else 'pts_r'
    img=L if side=='View 1' else R
    # Crop for usability, but ALWAYS keep original-image coordinates in state.
    if zoom:
        h,w=img.shape[:2]
        if 'pentagon' in example.lower():
            x0,y0,x1,y1=int(.29*w),int(.17*h),int(.83*w),int(.88*h)
        elif 'journal' in example.lower():
            x0,y0,x1,y1=int(.31*w),int(.23*h),int(.82*w),int(.89*h)
        else:
            x0,y0,x1,y1=int(.29*w),int(.21*h),int(.82*w),int(.84*h)
    else:
        h,w=img.shape[:2]; x0,y0,x1,y1=0,0,w,h
    crop=img[y0:y1,x0:x1]
    transformed_points=[(x-x0,y-y0) for x,y in st.session_state[pts_key]
                        if x0<=x<x1 and y0<=y<y1]
    screen=marked(crop,transformed_points)
    # Streamlit image_coordinates returns coordinates in display-image pixels.
    max_display_width=960
    if screen.width>max_display_width:
        display_scale=max_display_width/screen.width
        screen=screen.resize((max_display_width,round(screen.height*display_scale)),Image.Resampling.LANCZOS)
    else:
        display_scale=1.0
    click=streamlit_image_coordinates(screen,key=f'click-{example}-{side}-{zoom}-{len(st.session_state[pts_key])}')
    if click is not None:
        new=(min(w-1,x0+round(float(click['x'])/display_scale)),
             min(h-1,y0+round(float(click['y'])/display_scale)))
        st.session_state[pts_key].append(new)
        st.rerun()
    cl,cr=st.columns(2)
    with cl:
        if st.button('Undo last point'):
            if st.session_state[pts_key]: st.session_state[pts_key].pop()
            st.rerun()
    with cr:
        if st.button('Clear BOTH point sets'):
            st.session_state['pts_l']=[];st.session_state['pts_r']=[];st.rerun()
    st.write('View 1:',st.session_state['pts_l'],'  View 2:',st.session_state['pts_r'])
    if st.button('Calculate Z and dimensions',type='primary'):
        try:
            if 'pose' not in st.session_state or st.session_state.get('pose_for')!=current:
                raise ValueError('Estimate camera pose for this pair and calibration setting first.')
            n={'Rectangle':4,'Circle':2,'Polygon':5}[shape]
            if len(st.session_state['pts_l'])!=n or len(st.session_state['pts_r'])!=n:
                raise ValueError(f'Select exactly {n} points in each view.')
            rotation,t=st.session_state['pose']
            xyz=triangulate_points(st.session_state['pts_l'],st.session_state['pts_r'],K,dist,rotation,t,baseline)
            result=measure_shape(xyz,shape)
            zs=xyz[:,2]
            if np.ptp(zs)>max(5.0,0.15*abs(float(np.mean(zs)))):
                st.error('Reconstruction quality warning: the selected object points have widely differing depths. Do not use these dimensions as validated measurements; check the pose, baseline, and correspondences.')
            st.subheader('Estimated reconstruction')
            st.metric('Mean Z distance',f"{result['mean_depth_cm']:.2f} cm")
            if shape=='Rectangle':
                a,b=st.columns(2)
                a.metric('Width',f"{result['width_cm']:.2f} cm")
                b.metric('Height',f"{result['height_cm']:.2f} cm")
            elif shape=='Circle': st.metric('Diameter',f"{result['diameter_cm']:.2f} cm")
            else: st.write('Every polygon edge (cm)',{f'Edge {i+1}':round(e,2) for i,e in enumerate(result['edge_lengths_cm'])})
            st.write('All 3D points [X, Y, Z] in first-camera coordinates (cm):',np.round(xyz,2))
            if not calibrated: st.error('EXPLORATORY ONLY: focal length was assumed, not calibrated.')
            st.warning('Baseline was estimated by hand. Report this uncertainty and validate against your separate reference measurements.')
        except Exception as e: st.error(str(e))

with room_tab:
    st.subheader('Task 3 · Room measurement using video frames')
    st.write('Upload either room video, or use one of the included recordings. Select two frames from the **same video**, estimate camera pose, then mark corresponding room points. This is a real feature-based reconstruction prototype, not a guaranteed complete floor plan.')
    room_choice=st.selectbox('Room video', ['Video 1 · Module7_Room_Video_1.mp4','Video 2 · Module7_Room_Video_2.mp4','Upload my own video'])
    video_file=None
    if room_choice.startswith('Upload'):
        video_file=st.file_uploader('Upload MP4 or MOV',type=['mp4','mov','m4v'],key='room_upload')
        if video_file is None:
            st.info('Upload a video to begin; the preloaded recordings are also available.')
    else:
        video_path=BASE/'room_videos'/('Module7_Room_Video_1.mp4' if 'Video 1' in room_choice else 'Module7_Room_Video_2.mp4')
        if not video_path.exists(): st.error('Sample recording missing from deployed repository.')
    if video_file is not None:
        # Reuse a local temporary file only when its identity changes.
        signature=(video_file.name,len(video_file.getvalue()))
        if st.session_state.get('room_upload_signature')!=signature:
            old=st.session_state.get('room_uploaded_path')
            if old and os.path.exists(old): os.unlink(old)
            with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
                tmp.write(video_file.getvalue());st.session_state['room_uploaded_path']=tmp.name
            st.session_state['room_upload_signature']=signature
        video_path=Path(st.session_state['room_uploaded_path'])
    if (room_choice!='Upload my own video' or video_file is not None) and video_path.exists():
        try:
            meta=video_info(video_path)
            st.caption(f"Original video: {meta['width']} × {meta['height']} · {meta['fps']:.1f} FPS · {meta['duration_s']:.1f} seconds")
            a,b=st.columns(2)
            end=max(0.2,float(meta['duration_s'])-0.15)
            with a:
                time1=st.slider('Frame 1 timestamp (seconds)',0.0,end, min(2.0,end),0.1,key='room_t1')
            with b:
                time2=st.slider('Frame 2 timestamp (seconds)',0.0,end,min(5.0,end),0.1,key='room_t2')
            if abs(time2-time1)<0.4: st.warning('Choose frames at least 0.4 seconds apart with sufficient overlapping visual content.')
            F1,_=frame_at(video_path,time1)
            F2,_=frame_at(video_path,time2)
            if F1.shape!=F2.shape: raise ValueError('Frame sizes differ.')
            room_sig=(str(video_path),round(time1,2),round(time2,2))
            if st.session_state.get('room_sig')!=room_sig:
                st.session_state['room_sig']=room_sig
                st.session_state['room_pts1']=[];st.session_state['room_pts2']=[]
                st.session_state.pop('room_pose',None)
            qa,qb=st.columns(2)
            with qa: st.image(as_pil(F1),caption=f'Frame 1 · {time1:.1f}s',use_container_width=True)
            with qb: st.image(as_pil(F2),caption=f'Frame 2 · {time2:.1f}s',use_container_width=True)
            if 'camera_cal' in st.session_state:
                raw_K,raw_D,cal_size=st.session_state['camera_cal']
                vid_size=(F1.shape[1], F1.shape[0])
                if vid_size==cal_size:
                    K_room=np.array(raw_K,float); D_room=raw_D
                elif abs(vid_size[0]/vid_size[1]-cal_size[1]/cal_size[0]) < .015:
                    choice=st.selectbox('Checkerboard-to-video orientation', ['Clockwise 90°','Counterclockwise 90°'],key='room_cal_rotation')
                    rot_K,rot_D,rot_size=rotate_calibration_90(raw_K,raw_D,cal_size, 'clockwise' if choice.startswith('Clockwise') else 'counterclockwise')
                    K_room=scale_camera_matrix(rot_K,rot_size,vid_size); D_room=rot_D
                    st.info('Checkerboard intrinsics rotated and resized to video. Only valid if original capture used the same iPhone lens and no cropping or digital zoom; video export changes may invalidate this.')
                else:
                    st.error('Calibration photo aspect ratio is incompatible with video. Recalibrate using same camera mode and matching field of view.')
                    K_room=None;D_room=None
            else:
                K_room=None;D_room=None
                st.warning('Calibrate with the actual checkerboard in Tab 1 first. Uncalibrated video cannot provide validated dimensions.')
            if K_room is not None:
                if st.button('Estimate camera pose between selected video frames',type='primary'):
                    try:
                        R_room,t_room,stats_room=estimate_pose(F1,F2,K_room,D_room)
                        st.session_state['room_pose']=(R_room,t_room,stats_room)
                        st.success(f"{stats_room['pose_inliers']} geometric inliers from {stats_room['matches']} matched features. Camera displacement magnitude is still unknown.")
                    except Exception as e:
                        st.session_state.pop('room_pose',None);st.error(str(e))
                if 'room_pose' in st.session_state:
                    R_room,t_room,stats_room=st.session_state['room_pose']
                    st.write(f"Estimated camera translation direction [x,y,z] = {np.round(t_room,3).tolist()} (unitless).")
                    if abs(t_room[2])>.65:
                        st.warning('Movement has a large forward/backward component. Choose frames with better side-to-side parallax if possible.')
                    st.markdown('#### Mark matching room landmarks in the same order')
                    st.caption('Suggested: points 1–2 = endpoints of an independently measured reference (for scale); 3–4 = another room dimension to estimate. Add 5–6, 7–8 for further spans. Each point must represent the SAME physical location in both images.')
                    marked_side=st.radio('Mark points on', ['Frame 1','Frame 2'],horizontal=True,key='room_side')
                    pts_key='room_pts1' if marked_side=='Frame 1' else 'room_pts2'
                    display_frame=F1 if marked_side=='Frame 1' else F2
                    display_width=min(750,display_frame.shape[1])
                    scale=display_width/display_frame.shape[1]
                    shown=cv2.resize(display_frame,(display_width,round(display_frame.shape[0]*scale)))
                    pts_render=[(round(x*scale),round(y*scale)) for x,y in st.session_state[pts_key]]
                    click=streamlit_image_coordinates(marked(shown,pts_render),key=f'room_click_{marked_side}_{room_sig}',width=display_width)
                    click_key=f'{marked_side}_{room_sig}'
                    if click and st.session_state.get('last_room_click')!=(click_key,click['x'],click['y']):
                        st.session_state['last_room_click']=(click_key,click['x'],click['y'])
                        if len(st.session_state[pts_key])<8:
                            st.session_state[pts_key].append((click['x']/scale,click['y']/scale))
                            st.rerun()
                    k1,k2,k3=st.columns(3)
                    k1.write(f"Frame 1: {len(st.session_state['room_pts1'])} selected")
                    k2.write(f"Frame 2: {len(st.session_state['room_pts2'])} selected")
                    if k3.button('Clear room points'):
                        st.session_state['room_pts1']=[];st.session_state['room_pts2']=[];st.rerun()
                    method=st.radio('How will you establish metric scale?', ['Measured room reference between points 1–2','Measured camera displacement between THESE frames','No metric scale (relative units only)'],key='room_scale_method')
                    baseline_cm=None;reference_cm=None
                    if method.startswith('Measured room'):
                        reference_cm=st.number_input('Actual distance between marked points 1 and 2 (cm)',min_value=0.1,value=100.0,step=1.0,help='Enter the actual measured length of the exact two landmarks you clicked. For example, 6 feet 4 inches = 193.04 cm, but only if those are the correct endpoints.')
                        st.caption('The reference dimension is used for scaling, so it is NOT an independent accuracy check.')
                    elif method.startswith('Measured camera'):
                        baseline_cm=st.number_input('Lens-center displacement between these exact two frames (cm)',min_value=0.1,value=15.0,step=1.0,help='Do not use the 177.8 cm entire-video path unless these are precisely its endpoints and this is actual straight-line lens displacement.')
                    if st.button('Triangulate landmarks and estimate room spans'):
                        try:
                            P1=st.session_state['room_pts1'];P2=st.session_state['room_pts2']
                            X,sc,source=reconstruct_matched_points(P1,P2,K_room,D_room,R_room,t_room,baseline_cm,reference_cm)
                            st.subheader('Reconstructed landmarks')
                            st.dataframe(np.round(X,2))
                            if sc is None:
                                st.warning('No metric scale: the coordinates are relative, not centimeters. Choose an independent measured reference to obtain physical size.')
                            else:
                                st.success(f'Metric scale supplied by: {source}. All units shown in centimeters.')
                                for i in range(0,len(X)-1,2):
                                    label='Reference span (used for scale)' if i==0 and reference_cm is not None else f'Room span from marked points {i+1}–{i+2}'
                                    st.metric(label, f'{point_pair_distance(X,i,i+1):.2f} cm')
                                if np.ptp(X[:,2]) > 0.3*max(1,np.mean(X[:,2])):
                                    st.warning('Landmark depths vary widely. Check for mismatched points or poor camera geometry before trusting room dimensions.')
                            st.warning('Only measured spans are reconstructed: a complete room length × width × height requires endpoints for all three, and an independently verified scale. Occluded corners cannot be inferred reliably.')
                        except Exception as e: st.error(str(e))
            st.markdown('#### Independent room measurements (comparison targets only)')
            st.write('From iPhone Measure screenshots: room lengths mentioned approximately 12 ft 7 in and 11 ft; height approximately 9 ft 10 in. Another segment was 6 ft 4 in. Confirm exactly which wall/segment each measurement refers to before comparing.')
            st.caption('These reference values are not substituted for video reconstruction outputs. A single 1.78 m walking distance is not necessarily the straight camera displacement between chosen frames.')
        except Exception as e: st.error(f'Room-video analysis error: {e}')

with deriv_tab:
    st.subheader('Task 2 · Handwritten derivation')
    st.write('The professor requires **handwritten, scanned derivations** of UNCALIBRATED stereo. This app does not replace those pages.')
    st.write('Outline: correspondence → fundamental matrix and epipolar constraint → canonical projective cameras → triangulation → metric ambiguity → additional constraints necessary for real-world size.')
    st.warning('Pure uncalibrated stereo only reconstructs up to an arbitrary projective transform. Absolute physical size requires added calibration, geometric constraints or known reference scale.')
