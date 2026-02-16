IDE에서 클라우드 모드를 쓰려면, 먼저 Codex Cloud 환경을 만들어야 합니다.

1. IDE 확장에서 **ChatGPT 계정**으로 로그인하세요.

   Codex cloud는 ChatGPT 로그인만 지원하고, API 키 로그인만으로는 클라우드 모드를 못 씁니다.

2. 웹에서 `https://chatgpt.com/codex`로 가서 GitHub 연결을 완료하세요.

3. Codex 웹에서 **Environment 생성**하세요.

   저장소(repo) 선택 후 `Create environment`를 누르면 됩니다.

4. 환경 설정을 채우세요.

   자동 의존성 설치(일반 패키지 매니저 지원) 또는 수동 setup script, 환경변수/시크릿을 설정합니다.

5. IDE로 돌아와 채팅 입력창에서 `/cloud` 실행하세요.

   그다음 `/cloud-environment`로 방금 만든 환경을 선택합니다.

6. 프롬프트 입력 후 `Run in the cloud`로 실행하면 됩니다.
