// 로컬 파일 업로드를 위한 JavaScript 코드
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 파일 업로드 기능을 위한 확장
app.registerExtension({
    name: "LocalFileUpload",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LocalFileUpload" || nodeData.name === "LocalImageUpload") {
            // 노드 위젯에 파일 업로드 버튼 추가
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = origOnNodeCreated?.apply(this, arguments);
                
                // 파일 업로드 버튼 추가
                this.addWidget("button", "파일 업로드", null, () => {
                    this.uploadFile(false);
                });

                // 폴더 업로드 버튼 추가
                this.addWidget("button", "폴더 업로드", null, () => {
                    this.uploadFile(true);
                });
                
                // 업로드된 파일/폴더 정보를 저장할 속성
                this.uploadedInfo = null;
                
                return r;
            };
            
            // 파일/폴더 업로드 메서드
            nodeType.prototype.uploadFile = function(isFolder) {
                const input = document.createElement('input');
                input.type = 'file';
                
                if (isFolder) {
                    input.webkitdirectory = true;
                    input.multiple = true; // 폴더 내의 여러 파일을 처리하기 위해
                } else {
                    input.multiple = false;
                    // 파일 타입에 따라 필터 설정
                    if (nodeData.name === "LocalImageUpload") {
                        input.accept = 'image/*';
                    } else {
                        input.accept = '*/*';
                    }
                }
                
                input.onchange = async (e) => {
                    const files = Array.from(e.target.files);
                    if (files.length === 0) return;
                    
                    try {
                        let totalSize = 0;
                        for (const file of files) {
                            totalSize += file.size;
                        }

                        if (totalSize === 0 && files.length > 0) {
                            app.ui.dialog.show("선택된 폴더가 비어있습니다.");
                            return;
                        }

                        // 여러 파일을 서버에 업로드
                        const results = await this.uploadFilesToServer(files);
                        
                        // 업로드된 정보 저장 (폴더의 경우 첫 번째 파일 정보를 기반으로 함)
                        this.uploadedInfo = results[0]; 
                        if (isFolder) {
                            // 폴더의 경우, 파일 경로가 아닌 폴더 경로를 저장해야 합니다.
                            const folderPath = results[0].folder_path;
                            this.uploadedInfo.file_path = folderPath;
                            this.uploadedInfo.content = `폴더 업로드 완료: ${folderPath}`;
                        }
                        
                        // 업로드 트리거 값 증가 (노드 재실행을 위해)
                        const uploadTriggerWidget = this.widgets.find(w => w.name === "upload_trigger");
                        if (uploadTriggerWidget) {
                            uploadTriggerWidget.value = (uploadTriggerWidget.value || 0) + 1;
                        }
                        
                        // 노드 새로고침
                        this.setDirtyCanvas(true, true);
                        
                        // 성공 메시지
                        const message = isFolder 
                            ? `폴더가 성공적으로 업로드되었습니다: ${this.uploadedInfo.file_path}`
                            : `파일이 성공적으로 업로드되었습니다: ${results[0].filename}`;
                        app.ui.dialog.show(message);
                        
                    } catch (error) {
                        console.error("파일 업로드 실패:", error);
                        app.ui.dialog.show("파일 업로드에 실패했습니다: " + error.message);
                    }
                };
                
                input.click();
            };
            
            // 서버에 여러 파일을 업로드하는 메서드
            nodeType.prototype.uploadFilesToServer = async function(files) {
                const formData = new FormData();
                
                // 서브폴더 정보 가져오기
                const subfolderWidget = this.widgets.find(w => w.name === "subfolder");
                const subfolder = subfolderWidget ? subfolderWidget.value || "" : "";
                
                if (subfolder) {
                    formData.append('subfolder', subfolder);
                }

                for (const file of files) {
                    formData.append('files', file, file.webkitRelativePath || file.name);
                }
                
                // [수정됨] 파일을 input 디렉토리에 저장 (절대 경로 사용)
                const response = await fetch(new URL('/upload/input_folder', window.location.origin).href, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
            };
            
            // 노드 실행 시 업로드된 파일 정보 반환
            const origExecute = nodeType.prototype.load_file;
            if (origExecute) {
                nodeType.prototype.load_file = function(...args) {
                    if (this.uploadedInfo) {
                        // 업로드된 파일 정보를 반환
                        return (this.uploadedInfo.content || "업로드 완료.", 
                                this.uploadedInfo.file_path || this.uploadedInfo.filename, 
                                this.uploadedInfo.file_size || 0);
                    }
                    return origExecute.apply(this, args);
                };
            }
        }
    }
});

// 파일 드래그 앤 드롭 기능 추가
app.registerExtension({
    name: "FileDragDrop",
    
    async setup() {
        // 워크스페이스에 드래그 앤 드롭 이벤트 리스너 추가
        const workspace = document.querySelector('#comfyui_workspace');
        
        if (workspace) {
            workspace.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                workspace.classList.add('drag-over');
            });
            
            workspace.addEventListener('dragleave', (e) => {
                e.preventDefault();
                workspace.classList.remove('drag-over');
            });
            
            workspace.addEventListener('drop', async (e) => {
                e.preventDefault();
                workspace.classList.remove('drag-over');
                
                const files = Array.from(e.dataTransfer.files);
                
                for (const file of files) {
                    try {
                        // 파일을 서버에 업로드
                        const result = await this.uploadFileToServer(file);
                        
                        // 새 파일 업로드 노드 생성
                        const node = LiteGraph.createNode("LocalFileUpload");
                        node.uploadedFile = result;
                        
                        // 업로드 트리거 값 설정
                        const uploadTriggerWidget = node.widgets.find(w => w.name === "upload_trigger");
                        if (uploadTriggerWidget) {
                            uploadTriggerWidget.value = 1;
                        }
                        
                        app.graph.add(node);
                        
                        // 성공 메시지
                        app.ui.dialog.show(`파일이 성공적으로 업로드되었습니다: ${result.filename}`);
                        
                    } catch (error) {
                        console.error("파일 업로드 실패:", error);
                        app.ui.dialog.show("파일 업로드에 실패했습니다: " + error.message);
                    }
                }
            });
        }
    },
    
    async uploadFileToServer(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // [수정됨] 절대 경로 사용
        const response = await fetch(new URL('/upload/input', window.location.origin).href, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
});

// CSS 스타일 추가
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        background-color: rgba(0, 123, 255, 0.1) !important;
        border: 2px dashed #007bff !important;
    }
    
    .comfy-file-upload-button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        margin: 4px;
        transition: all 0.3s ease;
    }
    
    .comfy-file-upload-button:hover {
        background: linear-gradient(45deg, #0056b3, #004085);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
    }
    
    .file-upload-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 8px;
        margin: 4px 0;
        font-size: 12px;
        color: #495057;
    }
`;
document.head.appendChild(style);