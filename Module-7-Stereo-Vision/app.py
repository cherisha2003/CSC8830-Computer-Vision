"""CSC 8830 Assignment 7, Task 1: calibrated stereo object measurement.

Run: streamlit run app.py
Use the Calibration tab first. Without calibration, estimates are explicitly
marked exploratory and do not satisfy the calibrated-stereo requirement.
"""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from stereo_geometry import decode_image, calibrate_single_camera, estimate_pose, triangulate_points, measure_shape, rotate_calibration_90

st.set_page_config(page_title='CSC 8830 · Stereo Measurement', layout='wide')
st.title('Module 7 · Calibrated Stereo Object Measurement')
st.caption('Task 1 · Camera depth (Z) first, then real-world size · Photographs captured with a single moving iPhone')
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

cal_tab, measure_tab, room_tab, deriv_tab = st.tabs(['1 · Camera calibration', '2 · Object measurement', '3 · Room video (plan)', '4 · Uncalibrated derivation (plan)'])

with cal_tab:
    st.subheader('Calibrate iPhone intrinsics')
    st.write('Capture **15–20 photos of a printed checkerboard** at different angles and positions with the SAME 1× lens and image resolution used for the object pairs. A checkerboard is a calibration target, not 20 objects.')
    a,b,c=st.columns(3)
    with a: nx=st.number_input('Inner corners across',min_value=3,max_value=20,value=9)
    with b: ny=st.number_input('Inner corners down',min_value=3,max_value=20,value=6)
    with c: square_cm=st.number_input('Printed square side (cm)',min_value=0.1,value=2.0,step=0.1)
    uploads=st.file_uploader('Checkerboard images (8 minimum, 15–20 recommended)',type=['jpg','jpeg','png'],accept_multiple_files=True)
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
    st.subheader('Task 3 · Room dimensions from video')
    st.write('This section is a **capture and implementation plan**, NOT a finished room-measurement algorithm. Record a slow overlapping room video and independently measure a reference distance; a single ordinary video has unknown metric scale.')
    st.file_uploader('Room video — for next implementation stage',type=['mp4','mov'])
    st.info('Planned method: camera calibration → keyframe correspondences / camera poses → scale via measured reference → reconstruct visible room corners → compute length, width and height → compare with tape measurements.')

with deriv_tab:
    st.subheader('Task 2 · Handwritten derivation')
    st.write('The professor requires **handwritten, scanned derivations** of UNCALIBRATED stereo. This app does not replace those pages.')
    st.write('Outline: correspondence → fundamental matrix and epipolar constraint → canonical projective cameras → triangulation → metric ambiguity → additional constraints necessary for real-world size.')
    st.warning('Pure uncalibrated stereo only reconstructs up to an arbitrary projective transform. Absolute physical size requires added calibration, geometric constraints or known reference scale.')
