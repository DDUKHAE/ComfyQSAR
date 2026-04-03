import os
import pandas as pd
import multiprocessing
import traceback
import tempfile
import folder_paths
from padelpy import padeldescriptor

def build_padel_options(descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
                        use_file_name_as_molname, retain_order, threads, waiting_jobs,
                        max_runtime, max_cpd_per_file, headless, log):
    return {
        "fingerprints": False,
        "d_2d": descriptor_type,
        "d_3d": not descriptor_type,
        "detectaromaticity": detect_aromaticity,
        "removesalt": remove_salt,
        "standardizenitro": standardize_nitro,
        "usefilenameasmolname": use_file_name_as_molname,
        "retainorder": retain_order,
        "threads": threads,
        "waitingjobs": waiting_jobs,
        "maxruntime": max_runtime,
        "maxcpdperfile": max_cpd_per_file,
        "headless": headless,
        "log": log,
    }

class Descriptor_Calculations_Regression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "standardized_data": ("STRING", {"tooltip": "Path to standardized CSV (SMILES + value)"}),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": ("BOOLEAN", {"default": True, "label_on": "2D", "label_off": "3D"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True}),
                "remove_salt": ("BOOLEAN", {"default": True}),
                "standardize_nitro": ("BOOLEAN", {"default": True}),
                "log": ("BOOLEAN", {"default": False}),
                "use_file_name_as_molname": ("BOOLEAN", {"default": False}),
                "retain_order": ("BOOLEAN", {"default": True}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000}),
                "headless": ("BOOLEAN", {"default": True}),
                "threads": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS",)
    FUNCTION = "calculate_descriptors"
    CATEGORY = "QSAR/REGRESSION"
    OUTPUT_NODE = True

    def calculate_descriptors(self, standardized_data, advanced, descriptor_type=True,
                               detect_aromaticity=True, remove_salt=True, standardize_nitro=True,
                               log=False, use_file_name_as_molname=False, retain_order=True,
                               max_runtime=10000, max_cpd_per_file=0, headless=True,
                               threads=-1, waiting_jobs=-1):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR/Descriptor_Calculation_Regression")
        os.makedirs(output_dir, exist_ok=True)
        try:
            df = pd.read_csv(standardized_data)
            smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
            value_col = next((c for c in df.columns if c.lower() == 'value'), None)
            if smiles_col is None or value_col is None:
                raise ValueError("Input CSV must have 'SMILES' and 'value' columns.")
            smiles_list = df[smiles_col].dropna().astype(str).tolist()
            values = df[value_col].values

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi', newline='') as tmp:
                tmp.write('\n'.join(smiles_list))
                smi_path = tmp.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_out:
                desc_path = tmp_out.name

            padel_options = build_padel_options(
                descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
                use_file_name_as_molname, retain_order, threads, waiting_jobs,
                max_runtime, max_cpd_per_file, headless, log
            )
            padel_options['mol_dir'] = smi_path
            padel_options['d_file'] = desc_path
            padeldescriptor(**padel_options)
            os.remove(smi_path)

            if os.path.exists(desc_path) and os.path.getsize(desc_path) > 0:
                df_desc = pd.read_csv(desc_path)
                os.remove(desc_path)
                if len(df_desc) == len(values):
                    df_desc['value'] = values
                else:
                    df_desc['value'] = values[:len(df_desc)]
                output_file = os.path.join(output_dir, "descriptors_with_values.csv")
                df_desc.to_csv(output_file, index=False)
                log_message = (
                    "========================================\n"
                    "🔹 Regression Descriptor Calculation Done! 🔹\n"
                    "========================================\n"
                    f"✅ Compounds: {len(df_desc)}\n"
                    f"🔢 Descriptors: {df_desc.shape[1] - 2}\n"
                    f"💾 Output File: {os.path.basename(output_file)}\n"
                    "========================================"
                )
                return {"ui": {"text": log_message}, "result": (str(output_file),)}
            else:
                raise ValueError("PaDEL descriptor calculation produced no output.")
        except Exception as e:
            return {"ui": {"text": f"❌ Error: {e}\n{traceback.format_exc()}"}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Regression": Descriptor_Calculations_Regression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Regression": "2. Descriptor Calculation",
}
