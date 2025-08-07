import gradio as gr
import asyncio
import os
import json
import base64
import requests
import fitz  # PyMuPDF
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.utils import setup_logger
import json_repair

# 设置日志
setup_logger("lightrag", level="INFO")

# 配置参数
WORKING_DIR = "./rag_data"
API_KEY = "sk-xxx"
BASE_URL = "http://xxx/v1/"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

class PDFAnalyzer:
    def __init__(self):
        self.lightrag_instance = None
        self.initialized = False
    
    async def initialize_rag(self):
        """初始化LightRAG实例"""
        if self.initialized:
            return self.lightrag_instance
            
        self.lightrag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                "qwen2.5-vl-7b-instruct",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=4096,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="Qwen/Qwen3-Embedding-8B",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                ),
            )
        )
        
        await self.lightrag_instance.initialize_storages()
        self.initialized = True
        return self.lightrag_instance
    
    def call_vision_api_with_base64(self, base64_image, question):
        """使用base64编码的图像调用视觉API"""
        api_url = f"{BASE_URL}chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        image_url = f"data:image/png;base64,{base64_image}"

        payload = {
            "model": "qwen2.5-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": question}
                    ]
                }
            ],
            "max_tokens": 2048
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def analyze_pdf(self, pdf_path, question):
        """分析PDF文件的每一页"""
        try:
            doc = fitz.open(pdf_path)
            all_results = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 将页面渲染为图像
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # 获取图像字节
                img_bytes = pix.tobytes("png")
                
                # Base64编码
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                
                # 调用API分析图像
                result = self.call_vision_api_with_base64(base64_image, question)
                all_results.append({
                    "page": page_num + 1,
                    "result": result
                })
            
            doc.close()
            return all_results
        except Exception as e:
            return {"error": f"处理PDF时出错: {str(e)}"}

    async def analyze_report_compliance(self, report_info):
        """使用LightRAG分析报告是否符合国家标准"""
        try:
            await self.initialize_rag()
            
            query = f"请判断这份报告是否符合国家标准，包括其中的每个指标是否都达到了国家标准的要求，并给出判断依据。\n{report_info}"
            
            mode = "hybrid"
            res = await self.lightrag_instance.aquery(
                query,
                param=QueryParam(mode=mode, only_need_context=False)
            )
            return res
        except Exception as e:
            return f"分析过程中出错: {str(e)}"

# 创建全局分析器实例
analyzer = PDFAnalyzer()

def process_pdf_file(file):
    """处理上传的PDF文件"""
    if file is None:
        return "请上传PDF文件", ""
    
    try:
        # 第一步：提取PDF信息
        question = "请详细分析这份拉伸测试报告，提取出产品的关键信息，比如产品型号、参数等，以及所有的关键数据，可能的维度包括但不限于最大力、屈服强度、抗拉强度、断后伸长率等，并以JSON格式返回。"
        
        results = analyzer.analyze_pdf(file.name, question)
        
        if isinstance(results, dict) and "error" in results:
            return f"PDF分析失败: {results['error']}", ""
        
        # 提取第一页的分析结果
        if results and len(results) > 0:
            first_result = results[0]["result"]
            if "choices" in first_result and len(first_result["choices"]) > 0:
                report_info = first_result["choices"][0]["message"]["content"]
                return "PDF信息提取完成", report_info
            else:
                return "PDF分析结果格式错误", ""
        else:
            return "未能获取PDF分析结果", ""
            
    except Exception as e:
        return f"处理PDF时出错: {str(e)}", ""

def analyze_compliance(report_info):
    """分析报告是否符合国家标准"""
    if not report_info.strip():
        return "请先上传并分析PDF报告"
    
    try:
        # 使用asyncio运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(analyzer.analyze_report_compliance(report_info))
            return result
        finally:
            loop.close()
    except Exception as e:
        return f"标准符合性分析失败: {str(e)}"

def create_pdf_analysis_interface():
    """创建PDF分析界面"""
    with gr.Blocks(title="PDF测试报告分析系统") as interface:
        gr.Markdown("# PDF测试报告分析系统")
        gr.Markdown("上传PDF测试报告，系统将分析其是否符合国家标准")
        
        with gr.Row():
            with gr.Column(scale=1):
                # PDF上传区域
                pdf_file = gr.File(
                    label="上传PDF测试报告", 
                    file_types=[".pdf"],
                    height=200
                )
                
                # 分析按钮
                analyze_btn = gr.Button("开始分析", variant="primary")
                
                # 状态显示
                status_text = gr.Textbox(
                    label="分析状态", 
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                # 提取的报告信息
                report_info = gr.Textbox(
                    label="提取的报告信息",
                    lines=15,
                    interactive=True,
                    placeholder="PDF分析结果将显示在这里..."
                )
        
        # 标准符合性分析区域
        with gr.Row():
            with gr.Column():
                compliance_btn = gr.Button("分析标准符合性", variant="secondary")
                
                compliance_result = gr.Textbox(
                    label="标准符合性分析结果",
                    lines=20,
                    interactive=False,
                    placeholder="标准符合性分析结果将显示在这里..."
                )
        
        # 事件绑定
        pdf_file.change(
            fn=process_pdf_file,
            inputs=[pdf_file],
            outputs=[status_text, report_info]
        )
        
        analyze_btn.click(
            fn=process_pdf_file,
            inputs=[pdf_file],
            outputs=[status_text, report_info]
        )
        
        compliance_btn.click(
            fn=analyze_compliance,
            inputs=[report_info],
            outputs=[compliance_result]
        )
        
        # 示例说明
        gr.Markdown("""
        ## 使用说明
        1. 上传PDF格式的测试报告文件
        2. 系统自动提取报告中的关键信息（产品型号、测试数据等）
        3. 点击"分析标准符合性"按钮，系统将基于国家标准数据库判断报告是否符合要求
        4. 查看详细的分析结果和判断依据
        
        ## 注意事项
        - 支持的文件格式：PDF
        - 建议上传清晰的扫描版或原生PDF文件
        - 分析过程可能需要几秒钟时间，请耐心等待
        """)
    
    return interface

if __name__ == "__main__":
    # 创建界面
    demo = create_pdf_analysis_interface()
    
    # 启动应用
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        debug=True
    )