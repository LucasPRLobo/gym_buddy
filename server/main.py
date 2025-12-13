#!/usr/bin/env python3
"""
FastAPI server scaffold for Gym Buddy API.
Implements core routes per docs/api/openapi.yaml with stubs/placeholders.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from uuid import uuid4
from datetime import datetime


app = FastAPI(title="Gym Buddy API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Schemas (minimal Pydantic)
# -----------------------------
class CoachVoice(BaseModel):
    id: str
    name: str
    tone: Optional[str] = None
    locale: Optional[str] = None
    persona_tags: List[str] = []
    supports_audio: bool = True
    supports_video: bool = False


class Profile(BaseModel):
    id: str
    sports: List[str] = []
    goals: Optional[str] = None
    coach_tone: Optional[str] = None
    default_coach_voice_id: Optional[str] = None
    per_sport_overrides: Dict[str, str] = {}
    solo_override: Optional[str] = None
    level: int = 1
    streak: int = 0
    xp: int = 0


class ProfileUpdate(BaseModel):
    sports: Optional[List[str]] = None
    goals: Optional[str] = None
    coach_tone: Optional[str] = None
    default_coach_voice_id: Optional[str] = None
    per_sport_overrides: Optional[Dict[str, str]] = None
    solo_override: Optional[str] = None


class Participant(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    coach_voice_id: Optional[str] = None


class JointDeviation(BaseModel):
    joint: str
    deviation: float


class SessionEvent(BaseModel):
    timestamp: datetime
    participant_id: str
    exercise_id: Optional[str] = None
    set_idx: Optional[int] = None
    reps: Optional[int] = None
    clip_url: Optional[str] = None
    quality: Optional[float] = None
    top_joints: Optional[List[JointDeviation]] = None
    tempo: Optional[str] = None
    feedback_text: Optional[str] = None
    coach_voice_id_used: Optional[str] = None
    mode: Optional[str] = None


class SessionCreate(BaseModel):
    type: str
    sport: str
    coach_voice_id: Optional[str] = None
    start_at: Optional[datetime] = None
    allow_personal_voice_override: bool = False
    planned_exercises: Optional[List[str]] = None


class Session(BaseModel):
    id: str
    type: str
    sport: str
    coach_voice_id: Optional[str] = None
    allow_personal_voice_override: bool = False
    participants: List[Participant] = []
    events: List[SessionEvent] = []


class GroupCreate(BaseModel):
    name: str
    sport: Optional[str] = None
    coach_voice_id: Optional[str] = None
    allow_member_override: bool = False


class Group(BaseModel):
    id: str
    name: str
    sport: Optional[str] = None
    coach_voice_id: Optional[str] = None
    allow_member_override: bool = False


class ScheduleEntry(BaseModel):
    session_id: Optional[str] = None
    sport: str
    coach_voice_id: Optional[str] = None
    type: str
    start_at: datetime


# -----------------------------
# In-memory stores (stub)
# -----------------------------
db_profiles: Dict[str, Profile] = {}
db_voices: Dict[str, CoachVoice] = {}
db_sessions: Dict[str, Session] = {}
db_groups: Dict[str, Group] = {}
db_schedule: Dict[str, List[ScheduleEntry]] = {}


def get_current_user_id():
    # Stub auth; replace with real auth integration
    return "user-123"


# -----------------------------
# Routes
# -----------------------------
@app.get("/profile", response_model=Profile)
def get_profile():
    uid = get_current_user_id()
    if uid not in db_profiles:
        db_profiles[uid] = Profile(id=uid)
    return db_profiles[uid]


@app.put("/profile", response_model=Profile)
def update_profile(payload: ProfileUpdate):
    uid = get_current_user_id()
    profile = db_profiles.get(uid, Profile(id=uid))
    data = profile.dict()
    upd = payload.dict(exclude_unset=True)
    data.update({k: v for k, v in upd.items() if v is not None})
    db_profiles[uid] = Profile(**data)
    return db_profiles[uid]


@app.get("/coach_voices", response_model=Dict[str, List[CoachVoice]])
def list_voices():
    if not db_voices:
        # Seed a couple voices
        db_voices["voice-arnold"] = CoachVoice(id="voice-arnold", name="Coach Arnold", tone="tough")
        db_voices["voice-lebron"] = CoachVoice(id="voice-lebron", name="Coach LeBron", tone="motivational")
    return {"voices": list(db_voices.values())}


@app.post("/sessions", response_model=Session, status_code=201)
def create_session(payload: SessionCreate):
    sid = str(uuid4())
    session = Session(
        id=sid,
        type=payload.type,
        sport=payload.sport,
        coach_voice_id=payload.coach_voice_id,
        allow_personal_voice_override=payload.allow_personal_voice_override,
        participants=[],
        events=[],
    )
    db_sessions[sid] = session
    return session


@app.get("/sessions/{id}", response_model=Session)
def get_session(id: str):
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.post("/sessions/{id}/join")
def join_session(id: str):
    uid = get_current_user_id()
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if not any(p.user_id == uid for p in sess.participants):
        sess.participants.append(Participant(user_id=uid, display_name=uid))
    return {"status": "joined"}


@app.post("/sessions/{id}/leave")
def leave_session(id: str):
    uid = get_current_user_id()
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.participants = [p for p in sess.participants if p.user_id != uid]
    return {"status": "left"}


@app.patch("/sessions/{id}/voice", response_model=Session)
def update_session_voice(id: str, body: Dict[str, str]):
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.coach_voice_id = body.get("coach_voice_id")
    return sess


@app.post("/sessions/{id}/participant_voice")
def set_participant_voice(id: str, body: Dict[str, str]):
    uid = get_current_user_id()
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    cv = body.get("coach_voice_id")
    for p in sess.participants:
        if p.user_id == uid:
            p.coach_voice_id = cv
            break
    return {"status": "saved"}


@app.post("/sessions/{id}/event")
def append_event(id: str, payload: SessionEvent):
    sess = db_sessions.get(id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess.events.append(payload)
    return {"status": "recorded"}


@app.post("/groups", response_model=Group, status_code=201)
def create_group(payload: GroupCreate):
    gid = str(uuid4())
    group = Group(
        id=gid,
        name=payload.name,
        sport=payload.sport,
        coach_voice_id=payload.coach_voice_id,
        allow_member_override=payload.allow_member_override,
    )
    db_groups[gid] = group
    return group


@app.patch("/groups/{id}/voice")
def update_group_voice(id: str, body: Dict[str, str]):
    grp = db_groups.get(id)
    if not grp:
        raise HTTPException(status_code=404, detail="Group not found")
    grp.coach_voice_id = body.get("coach_voice_id")
    return {"status": "updated"}


@app.get("/schedule")
def list_schedule():
    uid = get_current_user_id()
    return {"entries": db_schedule.get(uid, [])}


@app.post("/schedule", status_code=201)
def create_schedule_entry(payload: ScheduleEntry):
    uid = get_current_user_id()
    db_schedule.setdefault(uid, []).append(payload)
    return payload


@app.post("/media/sign")
def sign_media(body: Dict[str, str]):
    # Placeholder; replace with cloud storage signed URLs
    fname = body.get("filename", "upload.bin")
    url = f"https://storage.local/{fname}"
    return {"upload_url": url, "file_url": url}


@app.post("/infer/exercise")
def infer_exercise(body: Dict[str, str]):
    # Placeholder; integrate ST-GCN exercise classifier
    return {"exercise_id": "squat", "confidence": 0.9, "reps": None}


@app.post("/infer/quality")
def infer_quality(body: Dict[str, str]):
    # Placeholder; integrate contrastive quality model
    return {
        "quality_score": 78.0,
        "top_joints": [{"joint": "left_knee", "deviation": 0.3}],
        "tempo_tag": "slow",
    }


@app.post("/feedback")
def generate_feedback(body: Dict[str, object]):
    # Placeholder; integrate LLM
    return {"text": "Keep your chest up and drive through your heels.", "audio_url": None}


@app.get("/progress")
def get_progress():
    return {"xp": 1000, "level": 5, "streak": 3, "badges": []}


@app.get("/leaderboard")
def get_leaderboard(challenge_id: Optional[str] = None):
    return {"entries": [{"user_id": "user-123", "score": 100}]}


@app.get("/")
def root():
    return {"status": "ok"}
