.DEFAULT_GOAL := help
.PHONY: help fmt lint fix check install-hooks run_gateway run_stt run_tts run_tts_vieneu run_rvc run_llm run_ui run_all stop_all health transcribe_sample tts_vieneu_sample chat_sample convert_audio

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CONDA_ENV      ?= voice
FISH_CONDA_ENV ?= $(CONDA_ENV)

PY           := conda run --no-capture-output -n $(CONDA_ENV) python
RUFF         := conda run --no-capture-output -n $(CONDA_ENV) ruff
UVICORN      := conda run --no-capture-output -n $(CONDA_ENV) uvicorn
UVICORN_FISH := conda run --no-capture-output -n $(FISH_CONDA_ENV) uvicorn

GATEWAY_HOST ?= 127.0.0.1
GATEWAY_PORT ?= 8000

STT_HOST ?= 127.0.0.1
STT_PORT ?= 8001

TTS_HOST ?= 127.0.0.1
TTS_PORT ?= 8002

VBV_HOST ?= 127.0.0.1
VBV_PORT ?= 8005

RVC_HOST ?= 127.0.0.1
RVC_PORT ?= 8003

LLM_HOST ?= 127.0.0.1
LLM_PORT ?= 8004

UI_HOST ?= 0.0.0.0
UI_PORT ?= 7860

SAMPLE_AUDIO ?= data/chunks/speech_chunk_0001.wav

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
help:
	@echo ""
	@echo "Voice project — handy commands"
	@echo ""
	@echo "Audio preprocessing:"
	@echo "  make convert_audio   - batch convert mp3/m4a -> wav (audio/input -> audio/output)"
	@echo "  make convert_audio AUDIO_INPUT=my/dir AUDIO_OUTPUT=out/dir"
	@echo ""
	@echo "Code quality:"
	@echo "  make fmt           - format Python (ruff format)"
	@echo "  make lint          - lint Python (ruff check)"
	@echo "  make fix           - lint + autofix (ruff check --fix)"
	@echo "  make check         - fmt + fix"
	@echo "  make install-hooks - install git pre-commit hook (runs check before every commit)"
	@echo ""
	@echo "Run services:"
	@echo "  make run_all          - start STT + TTS(fish-speech) + LLM + Gateway"
	@echo "  make stop_all         - kill listeners on project ports (SIGKILL; OK if TTS is stuck)"
	@echo "  make run_gateway"
	@echo "  make run_stt"
	@echo "  make run_tts          - TTS fish-speech (conda voice_fish; TTS_ENGINES=fish)"
	@echo "  make run_tts_vieneu   - TTS VieNeu only (conda voice; TTS_ENGINES=vieneu)"
	@echo "  make run_vbv"
	@echo "  make run_llm          - LLM connector -> Anything-LLM :3001"
	@echo "  make run_ui           - Gradio UI (connects to Gateway)"
	@echo ""
	@echo "Quick smoke tests (requires gateway + relevant service running):"
	@echo "  make health"
	@echo "  make transcribe_sample"
	@echo "  make tts_vieneu_sample"
	@echo "  make chat_sample      - full STT->LLM->TTS pipeline"
	@echo ""
	@echo "Notes:"
	@echo "  - Override conda env: make CONDA_ENV=voice"
	@echo "  - Override sample audio: make SAMPLE_AUDIO=data/chunks/xxx.wav"
	@echo ""

# ------------------------------------------------------------
# Audio preprocessing
# ------------------------------------------------------------
AUDIO_INPUT  ?= audio/input
AUDIO_OUTPUT ?= audio/output

convert_audio:
	$(PY) scripts/convert_audio.py --input $(AUDIO_INPUT) --output $(AUDIO_OUTPUT)

# ------------------------------------------------------------
# Code quality
# ------------------------------------------------------------
fmt:
	$(RUFF) format services/ gateway/ scripts/ ui/ common/

lint:
	$(RUFF) check services/ gateway/ scripts/ ui/ common/

fix:
	$(RUFF) check services/ gateway/ scripts/ ui/ common/ --fix

check: fmt fix

install-hooks:
	cp scripts/hooks/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "pre-commit hook installed."

# ------------------------------------------------------------
# Run services
# ------------------------------------------------------------
run_gateway:
	$(UVICORN) gateway.app:app --host $(GATEWAY_HOST) --port $(GATEWAY_PORT)

run_stt:
	$(UVICORN) services.stt.app:app --host $(STT_HOST) --port $(STT_PORT)

run_tts:
	TTS_ENGINES=fish $(UVICORN_FISH) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT)

run_tts_vieneu:
	TTS_ENGINES=vieneu $(UVICORN) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT)

run_vbv:
	$(UVICORN) services.vbv.main:app --host $(VBV_HOST) --port $(VBV_PORT)

run_rvc:
	$(UVICORN) services.rvc.app:app --host $(RVC_HOST) --port $(RVC_PORT)

run_llm:
	$(UVICORN) services.llm.app:app --host $(LLM_HOST) --port $(LLM_PORT)

run_ui:
	$(PY) -m ui.app

run_all:
	@$(UVICORN) services.stt.app:app --host $(STT_HOST) --port $(STT_PORT) &
	@TTS_ENGINES=fish $(UVICORN_FISH) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT) &
	@$(UVICORN) services.vbv.main:app --host $(VBV_HOST) --port $(VBV_PORT) &
	@$(UVICORN) services.rvc.app:app --host $(RVC_HOST) --port $(RVC_PORT) &
	@$(UVICORN) services.llm.app:app --host $(LLM_HOST) --port $(LLM_PORT) &
	@$(UVICORN) gateway.app:app --host $(GATEWAY_HOST) --port $(GATEWAY_PORT)

# LISTEN sockets only; SIGKILL so stuck torch inference cannot ignore SIGTERM.
stop_all:
	@echo "stop_all: SIGKILL listeners on :$(STT_PORT) :$(TTS_PORT) :$(VBV_PORT) :$(RVC_PORT) :$(LLM_PORT) :$(GATEWAY_PORT) :$(UI_PORT)"
	@for round in 1 2; do \
		for port in $(STT_PORT) $(TTS_PORT) $(VBV_PORT) $(RVC_PORT) $(LLM_PORT) $(GATEWAY_PORT) $(UI_PORT); do \
			pids=$$(lsof -nP -iTCP:$$port -sTCP:LISTEN -t 2>/dev/null); \
			if [ -n "$$pids" ]; then \
				kill -9 $$pids 2>/dev/null || true; \
				echo "  killed :$$port pids=$$(echo $$pids | tr '\n' ' ')"; \
			fi; \
		done; \
		[ $$round -eq 1 ] && sleep 1; \
	done
	@echo "stop_all: done."

# ------------------------------------------------------------
# Smoke tests (curl)
# ------------------------------------------------------------
health:
	@resp=$$(curl -sS http://$(GATEWAY_HOST):$(GATEWAY_PORT)/health || true); \
	if printf "%s" "$$resp" | $(PY) -c "import sys, json; json.load(sys.stdin); print('ok')" >/dev/null 2>&1; then \
		printf "%s" "$$resp" | $(PY) -m json.tool; \
	else \
		echo "Health endpoint did not return JSON. Raw response:"; \
		curl -i http://$(GATEWAY_HOST):$(GATEWAY_PORT)/health || true; \
	fi

transcribe_sample:
	@curl -s -X POST http://$(GATEWAY_HOST):$(GATEWAY_PORT)/transcribe \
		-F "audio=@$(SAMPLE_AUDIO)" | $(PY) -m json.tool

tts_vieneu_sample:
	@curl -s -X POST http://$(GATEWAY_HOST):$(GATEWAY_PORT)/tts/vieneu \
		-F "text=Xin chào, hôm nay trời đẹp quá!" \
		--output data/output_make_vieneu.wav
	@$(PY) -c "import soundfile as sf; i=sf.info('data/output_make_vieneu.wav'); print(f'WAV saved: data/output_make_vieneu.wav | {i.samplerate}Hz | {i.duration:.2f}s | ch={i.channels}')"

chat_sample:
	@curl -s -X POST http://$(GATEWAY_HOST):$(GATEWAY_PORT)/chat \
		-F "audio=@$(SAMPLE_AUDIO)" \
		--output data/response.wav
	@$(PY) -c "import soundfile as sf; i=sf.info('data/response.wav'); print(f'Response audio: data/response.wav | {i.samplerate}Hz | {i.duration:.2f}s | ch={i.channels}')"

