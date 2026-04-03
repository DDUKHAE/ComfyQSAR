import os
import pandas as pd
import multiprocessing
import traceback
import tempfile
import folder_paths
from rdkit import Chem
from padelpy import padeldescriptor

def calculate_descriptors_from_file(input_path, options):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp_desc_file:
        desc_output_path = tmp_desc_file.name

    options['d_file'] = desc_output_path

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.lower().endswith('.sdf'):
        options['mol_dir'] = input_path
        padeldescriptor(**options)

    elif input_path.lower().endswith(('.csv', '.smi')):
        try:
            df = pd.read_csv(input_path, skip_blank_lines=True)
            smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
            if smiles_col is None:
                if not df.empty:
                    smiles_col = df.columns[0]
                    df.rename(columns={df.columns[0]: 'SMILES'}, inplace=True)
                else:
                    raise ValueError(f"Input file {os.path.basename(input_path)} is empty or has no identifiable SMILES column.")
            else:
                df.rename(columns={smiles_col: 'SMILES'}, inplace=True)

            smiles_list = df['SMILES'].dropna().astype(str).tolist()
            if not smiles_list:
                raise ValueError(f"No valid SMILES strings found in {os.path.basename(input_path)} after filtering NaN values.")

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi', newline='') as tmp_smi_file:
                tmp_smi_file.write('\n'.join(smiles_list))
                smi_input_path = tmp_smi_file.name

            options['mol_dir'] = smi_input_path
            padeldescriptor(**options)
            os.remove(smi_input_path)

        except pd.errors.EmptyDataError:
            raise ValueError(f"Input file {os.path.basename(input_path)} is empty.")
        except Exception as e:
            raise ValueError(f"Failed to read SMILES from {os.path.basename(input_path)}. Error: {e}")
    else:
        raise ValueError(f"Unsupported file format: {os.path.basename(input_path)}. Supported: .sdf, .csv, .smi")

    if os.path.exists(desc_output_path) and os.path.getsize(desc_output_path) > 0:
        df_desc = pd.read_csv(desc_output_path)
        os.remove(desc_output_path)
        return df_desc.to_dict(orient='records')
    return []

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

class Descriptor_Calculations_Classification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_std": ("STRING", {"tooltip": "Path to positive compounds file (SDF/CSV/SMI)"}),
                "negative_std": ("STRING", {"tooltip": "Path to negative compounds file (SDF/CSV/SMI)"}),
                "advanced": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "descriptor_type": ("BOOLEAN", {"default": True, "label_on": "2D", "label_off": "3D",
                                                "tooltip": "Choose descriptor type: 2D (faster) or 3D (more detailed)"}),
                "detect_aromaticity": ("BOOLEAN", {"default": True, "tooltip": "Detect and handle aromatic structures"}),
                "remove_salt": ("BOOLEAN", {"default": True, "tooltip": "Remove salt components from molecules"}),
                "standardize_nitro": ("BOOLEAN", {"default": True, "tooltip": "Standardize nitro groups"}),
                "log": ("BOOLEAN", {"default": False, "tooltip": "Enable PaDEL-Descriptor's internal logging to a file"}),
                "use_file_name_as_molname": ("BOOLEAN", {"default": False,
                                                          "tooltip": "Use filename instead of SMILES as molecule name (primarily for SDF input)"}),
                "retain_order": ("BOOLEAN", {"default": True, "tooltip": "Keep original molecule order"}),
                "max_runtime": ("INT", {"default": 10000, "min": 1000, "max": 100000, "step": 1000,
                                        "tooltip": "Maximum calculation time per molecule (milliseconds)"}),
                "max_cpd_per_file": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1000,
                                             "tooltip": "Split large files for processing (0 = no splitting)"}),
                "headless": ("BOOLEAN", {"default": True,
                                         "tooltip": "Run PaDEL-Descriptor without GUI (recommended for servers)"}),
                "threads": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1,
                                    "tooltip": f"Number of CPU threads for PaDEL-Descriptor (-1 for all available, 1-{multiprocessing.cpu_count()})"}),
                "waiting_jobs": ("INT", {"default": -1, "min": -1, "max": multiprocessing.cpu_count(), "step": 1,
                                         "tooltip": "Number of concurrent PaDEL-Descriptor jobs in queue (-1 for auto)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTORS",)
    FUNCTION = "calculate_and_merge_descriptors"
    CATEGORY = "QSAR/CLASSIFICATION"
    OUTPUT_NODE = True

    def calculate_and_merge_descriptors(self, positive_std, negative_std, advanced, descriptor_type,
                                        detect_aromaticity, remove_salt, standardize_nitro,
                                        use_file_name_as_molname, retain_order, threads, waiting_jobs,
                                        max_runtime, max_cpd_per_file, headless, log):
        output_dir = os.path.join(folder_paths.get_output_directory(), "QSAR/Descriptor_Calculation")
        os.makedirs(output_dir, exist_ok=True)

        padel_options = build_padel_options(
            descriptor_type, detect_aromaticity, remove_salt, standardize_nitro,
            use_file_name_as_molname, retain_order, threads, waiting_jobs,
            max_runtime, max_cpd_per_file, headless, log
        )

        try:
            print(f"Calculating descriptors for positive compounds from: {os.path.basename(positive_std)}")
            pos_descriptors_list = calculate_descriptors_from_file(positive_std, padel_options.copy())
            df_positive = pd.DataFrame(pos_descriptors_list)
            if df_positive.empty:
                print(f"Warning: No descriptors calculated for positive compounds from {os.path.basename(positive_std)}.")
            df_positive['Label'] = 1

            print(f"Calculating descriptors for negative compounds from: {os.path.basename(negative_std)}")
            neg_descriptors_list = calculate_descriptors_from_file(negative_std, padel_options.copy())
            df_negative = pd.DataFrame(neg_descriptors_list)
            if df_negative.empty:
                print(f"Warning: No descriptors calculated for negative compounds from {os.path.basename(negative_std)}.")
            df_negative['Label'] = 0

            df_final = pd.concat([df_positive, df_negative], ignore_index=True)
            if df_final.empty:
                raise ValueError("No descriptors were calculated for either positive or negative compounds.")

            if 'Name' in df_final.columns:
                name_col = df_final.pop('Name')
                df_final.insert(0, 'Name', name_col)

            final_file = os.path.join(output_dir, "final_merged_descriptors.csv")
            df_final.to_csv(final_file, index=False)

            log_message = (
                "========================================\n"
                "🔹 Descriptor Calculation & Merge Done! 🔹\n"
                "========================================\n"
                f"✅ Positive Molecules: {len(df_positive)}\n"
                f"✅ Negative Molecules: {len(df_negative)}\n"
                f"📊 Total Molecules: {len(df_final)}\n"
                f"🔢 Total Descriptors: {df_final.shape[1] - 2} (excluding Name and Label)\n"
                f"💾 Output File: {os.path.basename(final_file)}\n"
                "========================================"
            )
            return {"ui": {"text": log_message}, "result": (str(final_file),)}

        except ReferenceError as e:
            if "Java JRE 6+ not found" in str(e):
                error_message = (
                    "❌ PaDEL-Descriptor Error: Java JRE Not Found!\n\n"
                    "PaDEL-Descriptor requires Java to run. Please install a Java Runtime Environment (JRE) or JDK (version 11 is recommended) on your system and restart ComfyUI.\n\n"
                    f"Original Error: {e}"
                )
                return {"ui": {"text": [error_message]}, "result": ("",)}
            raise e
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            error_message = f"❌ Error during descriptor calculation: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": [error_message]}, "result": ("",)}
        except Exception as e:
            error_message = f"❌ An unexpected error occurred during descriptor calculation: {e}\n\nTraceback:\n{traceback.format_exc()}"
            return {"ui": {"text": [error_message]}, "result": ("",)}

NODE_CLASS_MAPPINGS = {
    "Descriptor_Calculations_Classification": Descriptor_Calculations_Classification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Descriptor_Calculations_Classification": "2. Descriptor Calculation",
}
