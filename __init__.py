"""
ComfyQSAR __init__.py

Copyright (C) 2024 ComfyQSAR Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

디렉토리 이름에 공백/특수문자가 포함되어 있어
importlib.util 을 사용해 각 모듈을 동적으로 로드합니다.
"""

import os
import importlib.util
import traceback

# 이 파일의 위치 기준으로 py/ 폴더 경로를 구성
_BASE = os.path.dirname(os.path.abspath(__file__))
_PY   = os.path.join(_BASE, "py")

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(rel_path: str, *, base: str = None) -> None:
    """rel_path: py/ 에서의 상대 경로 (예: 'Classification/01 Data Load .../xxx.py')"""
    abs_path = os.path.join(base if base else _PY, rel_path)
    if not os.path.isfile(abs_path):
        print(f"[ComfyQSAR] ⚠️  Module not found, skipping: {rel_path}")
        return
    # 모듈 이름은 고유하면 되므로 경로에서 파생
    module_name = rel_path.replace(os.sep, ".").replace(" ", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        print(f"[ComfyQSAR] ✅ Loaded: {rel_path}")
    except Exception:
        print(f"[ComfyQSAR] ❌ Failed to load: {rel_path}")
        traceback.print_exc()


# =============================================================================
# Classification modules
# =============================================================================
_CL = "Classification"

_load(f"{_CL}/01 Data Load & Standardization/Data_Load_and_Standardization.py")
_load(f"{_CL}/02 Descriptor Calculation/Descriptor_Calculation.py")
_load(f"{_CL}/03 Descriptor Preprocessing/Descriptor_Preprocessing.py")
_load(f"{_CL}/04 Data Split/Data_Split.py")
_load(f"{_CL}/05 Descriptor Optimization/05-1 Filter-based Selection/Filter_Based_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/LASSO_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/Decision_Tree_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/Random_Forest_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/XGBoost_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/LightGBM_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/RFE_Selection.py")
_load(f"{_CL}/05 Descriptor Optimization/05-2 Model-based Selection/SelectFromModel_Selection.py")
_load(f"{_CL}/06 Descriptor Combination/Descriptor_Combination_MemoryE.py")
_load(f"{_CL}/07 Hyperparameter Tuning & Model Training/Hyperparameter_Tuning_and_Model_Training.py")
_load(f"{_CL}/08 Model Validation/Model_Validation.py")

# =============================================================================
# Regression modules
# =============================================================================
_REG = "Regression"

_load(f"{_REG}/01 Data Load & Standardization/Data_Load_and_Standardization.py")
_load(f"{_REG}/02 Descriptor Calculation/Descriptor_Calculation.py")
_load(f"{_REG}/03 Descriptor Preprocessing/Descriptor_Preprocessing.py")
_load(f"{_REG}/04 Data Split/Data_Split.py")
_load(f"{_REG}/05 Descriptor Optimization/05-1 Filter-based Selection/Filter_Based_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/LASSO_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/Decision_Tree_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/Random_Forest_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/XGBoost_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/LightGBM_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/RFE_Selection.py")
_load(f"{_REG}/05 Descriptor Optimization/05-2 Model-based Selection/SelectFromModel_Selection.py")
_load(f"{_REG}/06 Descriptor Combination/Descriptor_Combination.py")
_load(f"{_REG}/07 Hyperparameter Tuning & Model Training/Hyperparameter_Tuning_and_Model_Training.py")
_load(f"{_REG}/08 Model Validation/Model_Validation.py")

# =============================================================================
# Screener modules (py/ 루트에 위치)
# =============================================================================
_load("Screener/qsar_screener.py", base=_PY)
_load("Screener/custom_user_screener.py", base=_PY)

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
