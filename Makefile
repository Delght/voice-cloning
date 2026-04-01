.DEFAULT_GOAL := help
.PHONY: help fmt lint fix check run_gateway run_stt run_tts run_tts_fish run_tts_all run_rvc run_llm health transcribe_sample tts_vieneu_sample chat_sample

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CONDA_ENV ?= voice

PY         := conda run -n $(CONDA_ENV) python
RUFF       := conda run -n $(CONDA_ENV) ruff
UVICORN    := conda run -n $(CONDA_ENV) uvicorn

GATEWAY_HOST ?= 127.0.0.1
GATEWAY_PORT ?= 8000

STT_HOST ?= 127.0.0.1
STT_PORT ?= 8001

TTS_HOST ?= 127.0.0.1
TTS_PORT ?= 8002

RVC_HOST ?= 127.0.0.1
RVC_PORT ?= 8003

LLM_HOST ?= 127.0.0.1
LLM_PORT ?= 8004

SAMPLE_AUDIO ?= data/chunks/speech_chunk_0001.wav

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
help:
	@echo ""
	@echo "Voice project — handy commands"
	@echo ""
	@echo "Code quality:"
	@echo "  make fmt        - format Python (ruff format)"
	@echo "  make lint       - lint Python (ruff check)"
	@echo "  make fix        - lint + autofix (ruff check --fix)"
	@echo "  make check      - fmt + fix"
	@echo ""
	@echo "Run services (each target runs a blocking server):"
	@echo "  make run_gateway"
	@echo "  make run_stt"
	@echo "  make run_tts          - TTS with VieNeu (default, lightweight)"
	@echo "  make run_tts_fish     - TTS with fish-speech only (heavier)"
	@echo "  make run_tts_all      - TTS with both engines (uses more RAM)"
	@echo "  make run_rvc"
	@echo "  make run_llm          - LLM connector → Anything-LLM :3001"
	@echo ""
	@echo "Quick smoke tests (requires gateway + relevant service running):"
	@echo "  make health"
	@echo "  make transcribe_sample"
	@echo "  make tts_vieneu_sample"
	@echo "  make chat_sample      - full STT→LLM→TTS pipeline"
	@echo ""
	@echo "Notes:"
	@echo "  - Override conda env: make CONDA_ENV=voice"
	@echo "  - Override sample audio: make SAMPLE_AUDIO=data/chunks/xxx.wav"
	@echo ""

# ------------------------------------------------------------
# Code quality
# ------------------------------------------------------------
fmt:
	$(RUFF) format services/ gateway/ scripts/

lint:
	$(RUFF) check services/ gateway/ scripts/

fix:
	$(RUFF) check services/ gateway/ scripts/ --fix

check: fmt fix

# ------------------------------------------------------------
# Run services
# ------------------------------------------------------------
run_gateway:
	$(UVICORN) gateway.app:app --host $(GATEWAY_HOST) --port $(GATEWAY_PORT)

run_stt:
	$(UVICORN) services.stt.app:app --host $(STT_HOST) --port $(STT_PORT)

run_tts:
	$(UVICORN) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT)

run_tts_fish:
	TTS_ENGINES=fish $(UVICORN) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT)

run_tts_all:
	TTS_ENGINES=fish,vieneu $(UVICORN) services.tts.app:app --host $(TTS_HOST) --port $(TTS_PORT)

run_rvc:
	$(UVICORN) services.rvc.app:app --host $(RVC_HOST) --port $(RVC_PORT)

run_llm:
	$(UVICORN) services.llm.app:app --host $(LLM_HOST) --port $(LLM_PORT)

# ------------------------------------------------------------
# Smoke tests (curl)
# ------------------------------------------------------------
health:
	@curl -s http://$(GATEWAY_HOST):$(GATEWAY_PORT)/health | $(PY) -m json.tool

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

