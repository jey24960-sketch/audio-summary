import streamlit as st
import os
import subprocess
from pathlib import Path

st.set_page_config(page_title="강의 자동 정리", page_icon="🎙️")

st.title("🎙️ 강의 자동 정리 시스템")

st.markdown("녹음 파일 업로드 → 버튼 클릭 → 노션 자동 정리")

# 🔐 환경변수 체크
required_env = ["OPENAI_API_KEY", "NOTION_TOKEN", "NOTION_PARENT_PAGE_ID"]
missing = [k for k in required_env if not os.getenv(k)]

if missing:
    st.error(f"❌ 환경변수 설정 필요: {', '.join(missing)}")
    st.stop()

# 📁 업로드
uploaded_file = st.file_uploader(
    "녹음 파일 업로드",
    type=["m4a", "aac", "wav"]
)

if uploaded_file:
    os.makedirs("audio", exist_ok=True)

    filepath = os.path.join("audio", uploaded_file.name)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"파일 업로드 완료: {uploaded_file.name}")

    # 🚀 실행 버튼
    if st.button("🚀 실행하기"):
        with st.spinner("처리 중... (몇 분 걸릴 수 있음)"):
            result = subprocess.run(
                ["python", "main.py", "--audio-dir", "./audio"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("✅ 노션 업로드 완료!")
        else:
            st.error("❌ 실행 중 오류 발생")
            st.text(result.stderr)

# 📂 결과 확인
if Path("outputs").exists():
    st.markdown("---")
    st.subheader("📂 생성된 결과 파일")

    for root, dirs, files in os.walk("outputs"):
        for file in files:
            st.text(os.path.join(root, file))