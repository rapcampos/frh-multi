qlogin_debug:
	qlogin -A CVLABPJ -q debug -b 1 -l elapstim_req=01:00:00 -T openmpi -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda12.3.2 -V

qsub_gpu:
	qsub qsub.sh

run_notebook:
	@if [ -z "$(N)" ]; then \
		echo "Error: N variable is not set. Usage: make run_notebook N=path/to/notebook.ipynb"; \
		exit 1; \
	fi
	@N_PATH=$$(dirname "$(N)"); \
	N_BASE=$$(basename "$(N)" .ipynb); \
	SUFFIX=""; [ "$(P)" = "bf16" ] && SUFFIX="_unquantized"; \
	OUT_NAME="$(OUT)"; [ -z "$$OUT_NAME" ] && OUT_NAME="$$N_BASE$$SUFFIX.ipynb"; \
	OUTPUT_DIR="$$N_PATH/outputs"; \
	mkdir -p "$$OUTPUT_DIR"; \
	echo "Running notebook: $(N)$(if $(P), [QUANTIZATION=$(P)])"; \
	uv run papermill \
		$(N) \
		"$$OUTPUT_DIR/$$OUT_NAME" \
		$(if $(P),-p QUANTIZATION $(P)) \
		--log-output \
		--progress-bar

run:
	uv run make run_notebook N=10_multilang_safebench_ablation.ipynb

budget:
	/opt/nec/nqsv/bin/rbudgetcheck