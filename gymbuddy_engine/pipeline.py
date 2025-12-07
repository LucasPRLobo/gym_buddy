
from gymbuddy_engine.ingestion.video_io import load_video
from gymbuddy_engine.pose.pose_estimator import PoseEstimator
from gymbuddy_engine.analysis.squat_rules import analyze_squat
from gymbuddy_engine.visualization.overlay import generate_overlay_video
from gymbuddy_engine.visualization.ideal_render import generate_ideal_video
from gymbuddy_engine.coach.feedback import generate_coach_feedback

def process_exercise_video(video_path: str, exercise: str, coach_id: str):
    # 1. carregar vídeo
    video = load_video(video_path)

    # 2. extrair pose
    pose_estimator = PoseEstimator(model_name="mediapipe")
    pose_seq = pose_estimator.extract_pose(video, exercise=exercise)

    # 3. analisar execução (MVP: só squat)
    if exercise == "squat":
        analysis = analyze_squat(pose_seq)
    else:
        raise NotImplementedError

    # 4. gerar vídeos
    user_overlay_path = generate_overlay_video(video, pose_seq)
    ideal_motion_path = generate_ideal_video(pose_seq, exercise=exercise)

    # 5. feedback do coach
    feedback_text = generate_coach_feedback(analysis, coach_id=coach_id)

    # 6. retorno "pronto pro app"
    return {
        "analysis": analysis,
        "feedback_text": feedback_text,
        "user_overlay_video": user_overlay_path,
        "ideal_motion_video": ideal_motion_path,
    }
