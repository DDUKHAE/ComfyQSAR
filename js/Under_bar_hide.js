import { app } from "../../../scripts/app.js";

// 적용되는 노드 목록 정의 (ComfyQSAR_TEXT.js와 동일하게 유지)
const TARGET_NODES = Object.freeze([
  // Data Loader & Standardization
  "Data_Loader_Regression",
  "Standardization_Regression",
  "Load_and_Standardize_Regression",
  "Data_Loader_Classification",
  "Standardization_Classification",
  "Load_and_Standardize_Classification",

  // Descriptor Calculations
  "Descriptor_Calculations_Regression",
  "Descriptor_Calculations_Classification",

  // Descriptor Preprocessing
  "Replace_inf_with_nan_Regression",
  "Replace_inf_with_nan_Classification",
  "Remove_high_nan_compounds_Regression",
  "Remove_high_nan_compounds_Classification",
  "Remove_high_nan_descriptors_Regression",
  "Remove_high_nan_descriptors_Classification",
  "Impute_missing_values_Regression",
  "Impute_missing_values_Classification",
  "Descriptor_preprocessing_Regression",
  "Descriptor_preprocessing_Classification",

  // Data Split
  "QSARDataSplit_Regression",
  "QSARDataSplit_Classification",

  // Descriptor Optimization (Filter-based)
  "Remove_Low_Variance_Descriptors_Regression",
  "Remove_Low_Variance_Descriptors_Classification",
  "Remove_High_Correlation_Features_Regression",
  "Remove_High_Correlation_Features_Classification",
  "Descriptor_Optimization_Regression",
  "Descriptor_Optimization_Classification",

  // Feature Selection - Regression (Model-based)
  "LassoFeatureSelection",
  "DecisionTreeFeatureSelection",
  "RandomForestFeatureSelection",
  "XGBoostFeatureSelection",
  "LightGBMFeatureSelection",
  "SelectFromModelFeatureSelection",
  "RFEFeatureSelection",

  // Feature Selection - Classification (Model-based)
  "lasso_CL",
  "decision_tree_CL",
  "random_forest_CL",
  "xgb_CL",
  "lgb_CL",
  "rfe_CL",
  "select_from_model_CL",

  // Feature Combination
  "Regression_Feature_Combination_Search",
  "Feature_Combination_Search_Classification",

  // Grid Search & Model Training
  "Hyperparameter_Grid_Search_Regression",
  "Hyperparameter_Grid_Search_Classification",

  // Model Validation
  "Model_Validation_Regression",
  "Model_Validation_Classification",

  // Screener
  "QSARDBScreener",
  "QSARCustomUserScreener",
]);

app.registerExtension({
  name: "HIDE_UNDERSCORE",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (TARGET_NODES.includes(nodeData.name)) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);

        // 모든 위젯을 순회
        for (const widget of this.widgets) {
          // 원래 라벨 텍스트를 저장
          const originalLabel = widget.name;
          let tempLabel = originalLabel;

          // 모델 약자 숨기기 (예: rf_max_depth -> max_depth)
          const prefixes = ["rf_", "dt_", "lr_", "lasso_", "ridge_", "elastic_", "svm_", "xgb_", "lgb_"];
          for (const prefix of prefixes) {
            if (tempLabel.startsWith(prefix)) {
              tempLabel = tempLabel.substring(prefix.length);
              break;
            }
          }

          // '_' 문자를 공백으로 대체한 새 라벨 생성
          const newLabel = tempLabel.replace(/_/g, " ");

          // 위젯의 표시 이름 변경
          if (widget.label !== undefined) {
            widget.label = newLabel;
          } else {
            // Some specific widgets might just use string assignment differently, but typically ComfyUI supports .label or .name
            widget.name = originalLabel; // ensure original name doesn't change
            widget.label = newLabel;
          }

          // DOM 요소가 있는 경우 텍스트 업데이트
          if (widget.element) {
            const labelElement = widget.element.querySelector(".widget-label");
            if (labelElement) {
              labelElement.textContent = newLabel;
            }
          }
        }

        return result;
      };
    }
  },
});
