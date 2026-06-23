FILE_NAME := "environment"
ENV_NAME := "pt2121"

create_venv:
	mamba env create -f "$(FILE_NAME)".yml

create_venv_explicit:
	mamba env create -n "$(ENV_NAME)" -f "$(FILE_NAME)___result_explicit.txt"

remove_venv:
	mamba env remove -n "$(ENV_NAME)"

remove_dir:
	rm -rf ~/Dev/mambaforge/envs/"$(ENV_NAME)"/

export_venv:
	mamba env export -n "$(ENV_NAME)" > "$(FILE_NAME)___result.yml"
	mamba run -n "$(ENV_NAME)" pip freeze > "$(FILE_NAME)___result.txt"
	mamba run -n "$(ENV_NAME)" python -VV > "python_version.txt"
	conda list --explicit --md5 -n "$(ENV_NAME)" > "$(FILE_NAME)___result_explicit.txt"
