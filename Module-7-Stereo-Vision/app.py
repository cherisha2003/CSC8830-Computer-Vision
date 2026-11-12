        video_path = room_videos[room_choice]
        if not video_path.is_file():
            st.error(f'Missing bundled video: {video_path.name}. Check the room_videos folder in GitHub.')
        else:
            st.caption(f'Loaded file: {video_path.name} · {video_path.stat().st_size / (1024 * 1024):.2f} MB')
            st.video(str(video_path))
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
    if video_path is not None and video_path.is_file():
        try:
            meta=video_info(video_path)
            st.caption(f"Original video: {meta['width']} × {meta['height']} · {meta['fps']:.1f} FPS · {meta['duration_s']:.1f} seconds")
            a,b=st.columns(2)
            end = max(0.2, round(float(meta['duration_s']) - 0.15, 1))
            # Each recording gets separate timestamp controls and matching-point state.
            video_key = 'upload' if room_choice == 'Upload my own video' else ('video1' if room_choice.startswith('Video 1') else 'video2')
            defaults = (3.0, 6.0) if video_key == 'video1' else (2.0, 4.0)
            with a:
                time1 = st.slider('Frame 1 timestamp (seconds)', 0.0, end, min(defaults[0], end), 0.1, key=f'room_t1_{video_key}')
            with b:
                time2 = st.slider('Frame 2 timestamp (seconds)', 0.0, end, min(defaults[1], end), 0.1, key=f'room_t2_{video_key}')
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
