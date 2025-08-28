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
import time
import re

print("[启动] 所有导入完成", flush=True)

# 设置日志
print("[启动] 设置LightRAG日志...", flush=True)
setup_logger("lightrag", level="INFO")
print("[启动] 日志设置完成", flush=True)

# 配置参数
print("[启动] 加载配置参数...", flush=True)
WORKING_DIR = "./"
API_KEY = "sk-xxx"
BASE_URL = "http://127.0.0.1:10010/v1"
VL_MODEL = "qwen25-vl-72b"
EMBEDDING_MODEL = "qwen3-embedding-8b"

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
BACKOFF_FACTOR = 2  # 指数退避因子

print("[启动] 配置参数加载完成", flush=True)

def retry_api_call(max_retries=MAX_RETRIES, delay=RETRY_DELAY, backoff_factor=BACKOFF_FACTOR):
    """API重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    print(f"[重试] API调用尝试 {attempt + 1}/{max_retries}", flush=True)
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"[重试] API调用在第 {attempt + 1} 次尝试后成功", flush=True)
                    return result
                except Exception as e:
                    last_exception = e
                    current_delay = delay * (backoff_factor ** attempt)
                    print(f"[重试] API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}", flush=True)
                    
                    if attempt < max_retries - 1:
                        print(f"[重试] 等待 {current_delay} 秒后重试...", flush=True)
                        time.sleep(current_delay)
                    else:
                        print(f"[重试] 所有重试都失败，放弃API调用", flush=True)
            
            # 如果所有重试都失败，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator

def async_retry_api_call(max_retries=MAX_RETRIES, delay=RETRY_DELAY, backoff_factor=BACKOFF_FACTOR):
    """异步API重试装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    print(f"[重试] 异步API调用尝试 {attempt + 1}/{max_retries}", flush=True)
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        print(f"[重试] 异步API调用在第 {attempt + 1} 次尝试后成功", flush=True)
                    return result
                except Exception as e:
                    last_exception = e
                    current_delay = delay * (backoff_factor ** attempt)
                    print(f"[重试] 异步API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}", flush=True)
                    
                    if attempt < max_retries - 1:
                        print(f"[重试] 等待 {current_delay} 秒后重试...", flush=True)
                        await asyncio.sleep(current_delay)
                    else:
                        print(f"[重试] 所有异步重试都失败，放弃API调用", flush=True)
            
            # 如果所有重试都失败，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator

def extract_json_from_response(response_text):
    """从LLM响应中提取JSON内容"""
    try:
        print("[格式化] 开始提取JSON内容...", flush=True)
        
        # 尝试直接解析整个响应
        try:
            json_data = json.loads(response_text)
            print("[格式化] 直接解析JSON成功", flush=True)
            return json_data
        except:
            pass
        
        # 查找```json代码块
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                json_data = json.loads(json_str)
                print("[格式化] 从代码块提取JSON成功", flush=True)
                return json_data
            except:
                # 尝试使用json_repair修复
                try:
                    json_data = json_repair.loads(json_str)
                    print("[格式化] JSON修复后解析成功", flush=True)
                    return json_data
                except:
                    pass
        
        # 查找任何可能的JSON结构 - 更精确的匹配
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 嵌套JSON
            r'\{.*?\}',  # 简单JSON
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, response_text, re.DOTALL)
            for match in json_matches:
                if len(match) > 50:  # 过滤太短的匹配
                    try:
                        json_data = json.loads(match)
                        print("[格式化] 从模式匹配提取JSON成功", flush=True)
                        return json_data
                    except:
                        try:
                            json_data = json_repair.loads(match)
                            print("[格式化] JSON修复后解析成功", flush=True)
                            return json_data
                        except:
                            continue
        
        print("[格式化] 无法提取JSON，返回原文本", flush=True)
        return None
        
    except Exception as e:
        print(f"[格式化] JSON提取异常: {str(e)}", flush=True)
        return None

def safe_get_nested_value(data, path, default="未知"):
    """安全地获取嵌套字典的值"""
    try:
        for key in path:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data if data is not None else default
    except:
        return default

def format_test_data_html(json_data):
    """将测试数据格式化为HTML显示"""
    try:
        if not json_data or not isinstance(json_data, dict):
            return "无法解析测试数据"
        
        html_parts = []
        
        # 添加整体样式
        html_parts.append("""
        <style>
            .report-container { font-family: 'Microsoft YaHei', Arial, sans-serif; }
            .info-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
            .data-card { background: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .test-table { border-collapse: collapse; width: 100%; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .test-table th { background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 12px 8px; text-align: center; font-weight: bold; }
            .test-table td { padding: 10px 8px; text-align: center; border: 1px solid #ddd; }
            .test-table tr:nth-child(even) { background-color: #f8f9fa; }
            .test-table tr:hover { background-color: #e3f2fd; }
            .avg-card { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; padding: 12px; border-radius: 8px; margin: 8px 0; }
            .cv-card { background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%); color: white; padding: 12px; border-radius: 8px; margin: 8px 0; }
            .metric-item { display: inline-block; margin: 5px 15px 5px 0; font-weight: bold; }
            .section-title { color: #2d3436; margin: 20px 0 10px 0; font-size: 18px; font-weight: bold; border-left: 4px solid #74b9ff; padding-left: 10px; }
        </style>
        <div class="report-container">
        """)
        
        # 产品信息部分 - 智能适配不同的key名称
        product_info_keys = ["产品信息", "基本信息", "试验信息", "材料信息", "产品基本信息"]
        product_info = None
        for key in product_info_keys:
            if key in json_data:
                product_info = json_data[key]
                break
        
        if product_info and isinstance(product_info, dict):
            html_parts.append('<h3 class="section-title">📋 产品信息</h3>')
            html_parts.append('<div class="info-card">')
            
            # 定义常见字段的显示优先级和格式化
            priority_fields = [
                (["产品型号", "型号", "产品名称", "材料名称"], "🔧"),
                (["材料类型", "材料", "材质"], "🏗️"),  
                (["生产日期", "试验日期", "日期"], "📅"),
                (["试验温度", "温度"], "🌡️"),
                (["试验湿度", "湿度"], "💧"),
                (["送检单位", "委托单位", "单位"], "🏢"),
                (["试验员", "操作员"], "👨‍🔬"),
            ]
            
            displayed_fields = set()
            
            # 按优先级显示字段
            for field_variants, icon in priority_fields:
                for field in field_variants:
                    if field in product_info and field not in displayed_fields:
                        value = product_info[field]
                        if isinstance(value, dict):
                            html_parts.append(f'<div style="margin: 8px 0;"><strong>{icon} {field}:</strong></div>')
                            for sub_key, sub_value in value.items():
                                html_parts.append(f'<div style="margin-left: 20px;">• {sub_key}: <span style="color: #74b9ff;">{sub_value}</span></div>')
                        else:
                            html_parts.append(f'<div style="margin: 8px 0;"><strong>{icon} {field}:</strong> <span style="color: #74b9ff;">{value}</span></div>')
                        displayed_fields.add(field)
                        break
            
            # 显示其他未处理的字段
            for key, value in product_info.items():
                if key not in displayed_fields:
                    if isinstance(value, dict):
                        html_parts.append(f'<div style="margin: 8px 0;"><strong>📝 {key}:</strong></div>')
                        for sub_key, sub_value in value.items():
                            html_parts.append(f'<div style="margin-left: 20px;">• {sub_key}: <span style="color: #74b9ff;">{sub_value}</span></div>')
                    else:
                        html_parts.append(f'<div style="margin: 8px 0;"><strong>📝 {key}:</strong> <span style="color: #74b9ff;">{value}</span></div>')
            
            html_parts.append('</div>')
        
        # 测试数据部分 - 智能适配不同的结构
        test_data_keys = ["测试数据", "试验数据", "检测数据", "测量数据", "实验数据"]
        test_data = None
        for key in test_data_keys:
            if key in json_data:
                test_data = json_data[key]
                break
        
        if test_data and isinstance(test_data, dict):
            html_parts.append('<h3 class="section-title">📊 测试数据</h3>')
            
            # 详细数据表格 - 适配不同的字段名
            detail_keys = ["详细数据", "测试结果", "试验结果", "检测结果", "数据详情"]
            detail_data = None
            for key in detail_keys:
                if key in test_data and isinstance(test_data[key], list):
                    detail_data = test_data[key]
                    break
            
            if detail_data and len(detail_data) > 0:
                html_parts.append('<h4 style="color: #2d3436;">详细测试结果</h4>')
                html_parts.append('<table class="test-table">')
                
                # 表头
                first_item = detail_data[0]
                html_parts.append('<tr>')
                for key in first_item.keys():
                    display_key = key.replace('Num', '序号').replace('最大力', '最大力(N)').replace('抗拉强度', '抗拉强度(MPa)')
                    html_parts.append(f'<th>{display_key}</th>')
                html_parts.append('</tr>')
                
                # 数据行
                for item in detail_data:
                    html_parts.append('<tr>')
                    for value in item.values():
                        # 格式化数值显示
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                        else:
                            formatted_value = str(value)
                        html_parts.append(f'<td>{formatted_value}</td>')
                    html_parts.append('</tr>')
                
                html_parts.append('</table>')
            
            # 平均值 - 适配不同的字段名
            avg_keys = ["平均值", "均值", "平均", "Average", "平均结果"]
            avg_data = None
            for key in avg_keys:
                if key in test_data and isinstance(test_data[key], dict):
                    avg_data = test_data[key]
                    break
            
            if avg_data:
                html_parts.append('<h4 style="color: #2d3436;">平均值</h4>')
                html_parts.append('<div class="avg-card">')
                html_parts.append('<div style="display: flex; flex-wrap: wrap; align-items: center;">')
                for key, value in avg_data.items():
                    display_key = key.replace('最大力', '最大力').replace('抗拉强度', '抗拉强度').replace('屈服强度', '屈服强度')
                    html_parts.append(f'<span class="metric-item">📈 {display_key}: {value}</span>')
                html_parts.append('</div></div>')
            
            # CV% - 适配不同的字段名
            cv_keys = ["CV%", "变异系数", "CV", "变异系数(%)", "离散系数"]
            cv_data = None
            for key in cv_keys:
                if key in test_data and isinstance(test_data[key], dict):
                    cv_data = test_data[key]
                    break
            
            if cv_data:
                html_parts.append('<h4 style="color: #2d3436;">变异系数 (CV%)</h4>')
                html_parts.append('<div class="cv-card">')
                html_parts.append('<div style="display: flex; flex-wrap: wrap; align-items: center;">')
                for key, value in cv_data.items():
                    display_key = key.replace('最大力', '最大力').replace('抗拉强度', '抗拉强度')
                    html_parts.append(f'<span class="metric-item">📊 {display_key}: {value}</span>')
                html_parts.append('</div></div>')
        
        # 其他信息
        processed_keys = set(["产品信息", "基本信息", "试验信息", "材料信息", "产品基本信息", 
                             "测试数据", "试验数据", "检测数据", "测量数据", "实验数据"])
        
        for key, value in json_data.items():
            if key not in processed_keys and value:
                html_parts.append(f'<h4 class="section-title">{key}</h4>')
                if isinstance(value, dict):
                    html_parts.append('<div class="data-card">')
                    for sub_key, sub_value in value.items():
                        html_parts.append(f'<p><strong>{sub_key}:</strong> {sub_value}</p>')
                    html_parts.append('</div>')
                else:
                    html_parts.append(f'<div class="data-card"><p>{value}</p></div>')
        
        html_parts.append('</div>')
        return "".join(html_parts)
        
    except Exception as e:
        print(f"[格式化] HTML格式化异常: {str(e)}", flush=True)
        return f"""
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
            <h4 style="color: #d63031; margin: 0 0 10px 0;">⚠️ 格式化出错</h4>
            <p>无法正确解析测试报告格式，原始内容如下：</p>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;">{str(json_data)[:1000]}...</pre>
        </div>
        """

@retry_api_call(max_retries=MAX_RETRIES)
def format_compliance_result(raw_result, report_json):
    """调用LLM格式化标准符合性分析结果"""
    print("[格式化] 开始格式化符合性分析结果...", flush=True)
    
    # 提取产品信息用于模板
    product_type = "钢板"
    material_name = "未知"
    thickness = "未知"
    
    if report_json and isinstance(report_json, dict):
        product_info = report_json.get("产品信息", {})
        if "材料类型" in product_info:
            product_type = product_info["材料类型"]
        if "材料名称" in product_info:
            thickness = product_info["材料名称"]
        if "试验类型" in product_info and "管" in product_info["试验类型"]:
            product_type = "钢管"
    
    format_prompt = f"""
请将以下标准符合性分析结果格式化为规范的结论报告。

原始分析结果：
{raw_result}

请按照以下模板格式化：

如果符合标准，使用模板：
"通过本次拉伸试验，测定了本批次{product_type}各项力学性能指标，结果表明{product_type} {thickness} 的各项指标符合相关中国标准要求：
• 最大力指标符合：[标准号] [标准名] - [具体要求]
• 抗拉强度指标符合：[标准号] [标准名] - [具体要求]  
• 屈服强度指标符合：[标准号] [标准名] - [具体要求]
• 弹性模量符合：[标准号] [标准名] - [具体要求]
• 断后伸长率符合：[标准号] [标准名] - [具体要求]"

如果不符合标准，使用模板：
"通过本次拉伸试验，测定了本批次{product_type}各项力学性能指标，结果表明{product_type} {thickness} 的部分指标不符合相关中国标准要求：
• [不符合的指标名称]不符合：[标准号] [标准名] - [要求vs实际值]
• [其他不符合项...]
• 符合的指标：[列出符合的指标]"

请保持简洁明了，突出关键信息。
"""
    
    api_url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": VL_MODEL,
        "messages": [
            {
                "role": "user", 
                "content": format_prompt
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.1
    }
    
    print("[格式化] 发送格式化请求到API...", flush=True)
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    if "error" in result:
        raise Exception(f"API返回错误: {result['error']}")
    
    if "choices" in result and len(result["choices"]) > 0:
        formatted_result = result["choices"][0]["message"]["content"]
        print("[格式化] 符合性结果格式化完成", flush=True)
        return formatted_result
    else:
        raise Exception("API响应格式错误")

def format_compliance_html(compliance_text):
    """将符合性分析结果格式化为HTML"""
    try:
        print("[格式化] 开始HTML格式化符合性结果...", flush=True)
        
        # 分析结果的不同部分
        lines = compliance_text.strip().split('\n')
        html_parts = []
        
        # 添加样式
        html_parts.append("""
        <div style="font-family: 'Microsoft YaHei', sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin: 10px 0;">
            <h3 style="color: white; margin: 0 0 15px 0; text-align: center; font-size: 18px; font-weight: bold;">
                🔍 标准符合性分析结果
            </h3>
        </div>
        <div style="padding: 20px; background: white; border: 2px solid #e0e0e0; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """)
        
        # 处理文本内容
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    html_parts.append(format_paragraph_html(paragraph_text))
                    current_paragraph = []
                continue
            
            current_paragraph.append(line)
        
        # 处理最后一个段落
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            html_parts.append(format_paragraph_html(paragraph_text))
        
        html_parts.append('</div>')
        
        result = "".join(html_parts)
        print("[格式化] 符合性HTML格式化完成", flush=True)
        return result
        
    except Exception as e:
        print(f"[格式化] HTML格式化异常: {str(e)}", flush=True)
        # 降级处理，至少保证可读性
        return f"""
        <div style="padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; font-family: 'Microsoft YaHei', sans-serif;">
            <h4 style="color: #495057; margin: 0 0 15px 0;">📋 分析结果</h4>
            <div style="line-height: 1.6; color: #212529; white-space: pre-line;">{compliance_text}</div>
        </div>
        """

def format_paragraph_html(paragraph_text):
    """格式化单个段落为HTML"""
    # 检查是否是结论性段落（通常包含"通过本次"、"结果表明"等关键词）
    if any(keyword in paragraph_text for keyword in ["通过本次", "结果表明", "符合相关", "不符合相关"]):
        # 这是主要结论段落，需要突出显示
        # 检查是否符合标准
        if "符合相关" in paragraph_text and "不符合" not in paragraph_text:
            # 符合标准 - 绿色主题
            bg_color = "#d4edda"
            border_color = "#c3e6cb" 
            text_color = "#155724"
            icon = "✅"
        else:
            # 不符合标准 - 红色主题
            bg_color = "#f8d7da"
            border_color = "#f5c6cb"
            text_color = "#721c24"
            icon = "❌"
        
        return f"""
        <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 15px; margin: 15px 0; position: relative;">
            <div style="position: absolute; top: -12px; left: 20px; background: white; padding: 0 10px; font-size: 20px;">{icon}</div>
            <h4 style="color: {text_color}; margin: 5px 0 10px 0; font-weight: bold;">综合结论</h4>
            <p style="color: {text_color}; line-height: 1.8; margin: 0; font-size: 15px;">{paragraph_text}</p>
        </div>
        """
    
    # 检查是否是列表项（包含 • 或者以指标名开头）
    elif "•" in paragraph_text or any(keyword in paragraph_text for keyword in ["指标", "强度", "符合", "要求"]):
        # 拆分为列表项
        if "•" in paragraph_text:
            items = [item.strip() for item in paragraph_text.split("•") if item.strip()]
        else:
            items = [paragraph_text]
        
        html = '<ul style="margin: 10px 0; padding-left: 0; list-style: none;">'
        for item in items:
            if not item:
                continue
            
            # 根据内容判断颜色
            if "符合" in item and "不符合" not in item:
                item_color = "#28a745"
                item_icon = "✓"
            elif "不符合" in item:
                item_color = "#dc3545"  
                item_icon = "✗"
            else:
                item_color = "#6c757d"
                item_icon = "•"
                
            html += f"""
            <li style="background: #f8f9fa; border-left: 4px solid {item_color}; padding: 12px 15px; margin: 8px 0; border-radius: 0 6px 6px 0;">
                <span style="color: {item_color}; font-weight: bold; margin-right: 8px;">{item_icon}</span>
                <span style="color: #495057; line-height: 1.6;">{item}</span>
            </li>
            """
        html += '</ul>'
        return html
    
    # 普通段落
    else:
        return f"""
        <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff;">
            <p style="color: #495057; line-height: 1.7; margin: 0; font-size: 14px;">{paragraph_text}</p>
        </div>
        """

print("[启动] 检查工作目录...", flush=True)
if not os.path.exists(WORKING_DIR):
    print(f"[启动] 创建工作目录: {WORKING_DIR}", flush=True)
    os.makedirs(WORKING_DIR)
else:
    print(f"[启动] 工作目录已存在: {WORKING_DIR}", flush=True)

class PDFAnalyzer:
    def __init__(self):
        print("[启动] 初始化PDFAnalyzer...", flush=True)
        self.lightrag_instance = None
        self.initialized = False
        print("[启动] PDFAnalyzer初始化完成", flush=True)
    
    async def initialize_rag(self):
        """初始化LightRAG实例"""
        if self.initialized:
            print("[后台] LightRAG已初始化，直接返回实例", flush=True)
            return self.lightrag_instance
            
        print("[后台] 开始初始化LightRAG实例...", flush=True)
        print(f"[后台] 工作目录: {WORKING_DIR}", flush=True)
        print(f"[后台] 使用模型: {VL_MODEL}", flush=True)
        print(f"[后台] 嵌入模型: {EMBEDDING_MODEL}", flush=True)
        
        self.lightrag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                VL_MODEL,
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
                    model=EMBEDDING_MODEL,
                    api_key=API_KEY,
                    base_url=BASE_URL,
                ),
            )
        )
        
        print("[后台] 正在初始化存储系统...", flush=True)
        await self.lightrag_instance.initialize_storages()
        self.initialized = True
        print("[后台] LightRAG初始化完成", flush=True)
        return self.lightrag_instance
    
    @retry_api_call(max_retries=MAX_RETRIES)
    def call_vision_api_with_base64(self, base64_image, question):
        """使用base64编码的图像调用视觉API"""
        print("[后台] 正在调用视觉API...", flush=True)
        api_url = f"{BASE_URL}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        image_url = f"data:image/png;base64,{base64_image}"
        print(f"[后台] API URL: {api_url}", flush=True)
        print(f"[后台] 使用模型: {VL_MODEL}", flush=True)
        print(f"[后台] 图像大小: {len(base64_image)} 字符", flush=True)

        payload = {
            "model": VL_MODEL,
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

        print("[后台] 发送API请求...", flush=True)
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        print("[后台] API请求成功", flush=True)
        
        result = response.json()
        
        # 检查API响应是否包含错误
        if "error" in result:
            raise Exception(f"API返回错误: {result['error']}")
        
        return result

    def analyze_pdf(self, pdf_path, question):
        """分析PDF文件的每一页"""
        try:
            print(f"[后台] 开始分析PDF文件: {pdf_path}", flush=True)
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            print(f"[后台] PDF总页数: {total_pages}", flush=True)
            all_results = []
            
            for page_num in range(total_pages):
                print(f"[后台] 正在处理第 {page_num + 1}/{total_pages} 页...", flush=True)
                page = doc.load_page(page_num)
                
                # 将页面渲染为图像
                print(f"[后台] 第{page_num + 1}页: 开始渲染图像...", flush=True)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # 获取图像字节
                img_bytes = pix.tobytes("png")
                print(f"[后台] 第{page_num + 1}页: 图像大小 {len(img_bytes)} 字节", flush=True)
                
                # Base64编码
                print(f"[后台] 第{page_num + 1}页: 进行Base64编码...", flush=True)
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                
                # 调用API分析图像（带重试机制）
                print(f"[后台] 第{page_num + 1}页: 开始API分析...", flush=True)
                try:
                    result = self.call_vision_api_with_base64(base64_image, question)
                    print(f"[后台] 第{page_num + 1}页: 分析完成", flush=True)
                    
                    all_results.append({
                        "page": page_num + 1,
                        "result": result
                    })
                except Exception as api_error:
                    print(f"[后台] 第{page_num + 1}页: API调用最终失败: {str(api_error)}", flush=True)
                    # 继续处理其他页面，但记录错误
                    all_results.append({
                        "page": page_num + 1,
                        "result": {"error": f"API调用失败: {str(api_error)}"}
                    })
            
            doc.close()
            print(f"[后台] PDF分析完成，共处理 {total_pages} 页", flush=True)
            return all_results
        except Exception as e:
            error_msg = f"处理PDF时出错: {str(e)}"
            print(f"[后台] 错误: {error_msg}", flush=True)
            return {"error": error_msg}

    @async_retry_api_call(max_retries=MAX_RETRIES)
    async def analyze_report_compliance(self, report_info):
        """使用LightRAG分析报告是否符合国家标准"""
        print("[后台] 开始分析报告符合性...", flush=True)
        await self.initialize_rag()
        
        query = f"请判断这份报告是否符合国家标准，包括其中的每个指标是否都达到了国家标准的要求，并给出判断依据。\n{report_info}"
        print("[后台] 构建查询语句完成", flush=True)
        print(f"[后台] 查询内容长度: {len(query)} 字符", flush=True)
        
        mode = "hybrid"
        print(f"[后台] 使用查询模式: {mode}", flush=True)
        print("[后台] 正在向LightRAG发送查询请求...", flush=True)
        
        res = await self.lightrag_instance.aquery(
            query,
            param=QueryParam(mode=mode, only_need_context=False)
        )
        
        print("[后台] LightRAG查询完成", flush=True)
        print(f"[后台] 返回结果长度: {len(str(res))} 字符", flush=True)
        return res

# 创建全局分析器实例
print("[后台] 正在创建PDF分析器实例...", flush=True)
try:
    analyzer = PDFAnalyzer()
    print("[后台] PDF分析器实例创建成功", flush=True)
except Exception as e:
    print(f"[后台] 创建PDF分析器失败: {str(e)}", flush=True)
    raise


def process_pdf_file(file):
    """处理上传的PDF文件"""
    print("[后台] ========== 开始处理PDF文件 ==========", flush=True)
    
    # 立即输出以确认函数被调用
    import sys
    sys.stdout.flush()
    
    if file is None:
        print("[后台] 错误: 未上传文件", flush=True)
        return "请上传PDF文件", ""
    
    print(f"[后台] 接收到文件: {file.name}", flush=True)
    print(f"[后台] 文件对象类型: {type(file)}", flush=True)
    print(f"[后台] 文件是否存在: {hasattr(file, 'name')}", flush=True)
    
    # 检查文件路径是否存在
    import os
    if hasattr(file, 'name') and file.name:
        file_exists = os.path.exists(file.name)
        print(f"[后台] 文件路径存在: {file_exists}", flush=True)
        if not file_exists:
            print(f"[后台] 错误: 文件路径不存在 {file.name}", flush=True)
            return "文件路径不存在", ""
    
    try:
        # 第一步：提取PDF信息
        question = "请详细分析这份拉伸测试报告，提取出产品的关键信息，比如产品型号、参数等，以及所有的关键数据，可能的维度包括但不限于最大力、屈服强度、抗拉强度、断后伸长率等，并以JSON格式返回。"
        print("[后台] 分析问题设置完成", flush=True)
        
        print("[后台] 开始调用PDF分析器...", flush=True)
        results = analyzer.analyze_pdf(file.name, question)
        
        if isinstance(results, dict) and "error" in results:
            print(f"[后台] PDF分析失败: {results['error']}", flush=True)
            return f"PDF分析失败: {results['error']}", ""
        
        print("[后台] 开始提取分析结果...", flush=True)
        # 提取第一页的分析结果
        if results and len(results) > 0:
            print(f"[后台] 获得 {len(results)} 页分析结果", flush=True)
            
            # 查找第一个成功的结果
            successful_result = None
            for result_item in results:
                result = result_item["result"]
                if not isinstance(result, dict) or "error" not in result:
                    successful_result = result_item
                    break
                    
            if successful_result:
                result = successful_result["result"]
                if "choices" in result and len(result["choices"]) > 0:
                    raw_report_info = result["choices"][0]["message"]["content"]
                    print(f"[后台] 提取原始报告信息成功，长度: {len(raw_report_info)} 字符", flush=True)
                    
                    # 提取并格式化JSON内容
                    print("[后台] 开始格式化报告信息...", flush=True)
                    json_data = extract_json_from_response(raw_report_info)
                    
                    if json_data:
                        # 格式化为HTML显示
                        formatted_html = format_test_data_html(json_data)
                        print("[后台] 报告信息格式化完成", flush=True)
                        print("[后台] ========== PDF处理完成 ===========", flush=True)
                        
                        # 将JSON数据存储起来供后续使用
                        if not hasattr(analyzer, 'last_report_json'):
                            analyzer.last_report_json = {}
                        analyzer.last_report_json = json_data
                        
                        return "PDF信息提取完成", formatted_html
                    else:
                        # 如果无法提取JSON，返回原始格式化的文本
                        print("[后台] 无法提取JSON，返回原始内容", flush=True)
                        formatted_text = raw_report_info.replace('\n', '<br>').replace('```json', '<pre>').replace('```', '</pre>')
                        print("[后台] ========== PDF处理完成 ===========", flush=True)
                        return "PDF信息提取完成（原始格式）", formatted_text
                else:
                    print("[后台] 错误: PDF分析结果格式错误", flush=True)
                    return "PDF分析结果格式错误", ""
            else:
                # 所有页面都失败了
                error_summary = []
                for result_item in results:
                    if isinstance(result_item["result"], dict) and "error" in result_item["result"]:
                        error_summary.append(f"第{result_item['page']}页: {result_item['result']['error']}")
                
                error_msg = f"所有页面分析都失败了:\n" + "\n".join(error_summary[:3])  # 只显示前3个错误
                if len(error_summary) > 3:
                    error_msg += f"\n... 以及其他 {len(error_summary) - 3} 个错误"
                    
                print(f"[后台] 所有页面都失败: {error_msg}", flush=True)
                return error_msg, ""
        else:
            print("[后台] 错误: 未能获取PDF分析结果", flush=True)
            return "未能获取PDF分析结果", ""
            
    except Exception as e:
        error_msg = f"处理PDF时出错: {str(e)}"
        print(f"[后台] 异常: {error_msg}", flush=True)
        print(f"[后台] 异常类型: {type(e)}", flush=True)
        import traceback
        print(f"[后台] 异常堆栈: {traceback.format_exc()}", flush=True)
        return error_msg, ""

def analyze_compliance(report_info):
    """分析报告是否符合国家标准"""
    print("[后台] ========== 开始标准符合性分析 ==========", flush=True)
    if not report_info.strip():
        print("[后台] 错误: 报告信息为空", flush=True)
        return """
        <div style="padding: 20px; background: #e2e3e5; border: 2px solid #c6c8ca; border-radius: 8px; font-family: 'Microsoft YaHei', sans-serif; text-align: center;">
            <h4 style="color: #6c757d; margin: 0 0 10px 0;">📋 等待分析</h4>
            <p style="color: #6c757d; line-height: 1.6; margin: 0;">请先上传PDF文件并点击"开始分析"按钮，然后再进行标准符合性分析。</p>
        </div>
        """
    
    print(f"[后台] 报告信息长度: {len(report_info)} 字符", flush=True)
    
    try:
        print("[后台] 创建新的事件循环...", flush=True)
        # 使用asyncio运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            print("[后台] 开始执行异步分析任务...", flush=True)
            raw_result = loop.run_until_complete(analyzer.analyze_report_compliance(report_info))
            print(f"[后台] 原始分析完成，结果长度: {len(str(raw_result))} 字符", flush=True)
            
            # 格式化分析结果
            try:
                print("[后台] 开始格式化符合性分析结果...", flush=True)
                report_json = getattr(analyzer, 'last_report_json', None)
                formatted_result = format_compliance_result(str(raw_result), report_json)
                print("[后台] 符合性分析结果格式化完成", flush=True)
                
                # 将结果转换为HTML格式显示
                html_result = format_compliance_html(formatted_result)
                print("[后台] ========== 标准符合性分析完成 ===========", flush=True)
                return html_result
            except Exception as format_error:
                print(f"[后台] 格式化失败，返回原始结果: {str(format_error)}", flush=True)
                print("[后台] ========== 标准符合性分析完成 ===========", flush=True)
                # 原始结果也转换为HTML显示
                return format_compliance_html(str(raw_result))
        finally:
            print("[后台] 关闭事件循环", flush=True)
            loop.close()
    except Exception as e:
        error_msg = f"标准符合性分析失败: {str(e)}"
        print(f"[后台] 异常: {error_msg}", flush=True)
        # 如果是重试机制失败，提供更友好的错误信息
        if "重试" in str(e) or "API调用失败" in str(e):
            error_html = f"""
            <div style="padding: 20px; background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; font-family: 'Microsoft YaHei', sans-serif;">
                <h4 style="color: #721c24; margin: 0 0 10px 0;">⚠️ 服务暂时不可用</h4>
                <p style="color: #721c24; line-height: 1.6; margin: 0;">分析服务暂时不可用，请稍后重试。</p>
                <details style="margin-top: 15px;">
                    <summary style="color: #721c24; cursor: pointer;">查看详细错误信息</summary>
                    <pre style="background: #721c24; color: white; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">{error_msg}</pre>
                </details>
            </div>
            """
            return error_html
        
        # 其他错误的HTML格式化
        return f"""
        <div style="padding: 20px; background: #fff3cd; border: 2px solid #ffeaa7; border-radius: 8px; font-family: 'Microsoft YaHei', sans-serif;">
            <h4 style="color: #856404; margin: 0 0 10px 0;">❗ 分析过程出现问题</h4>
            <p style="color: #856404; line-height: 1.6; margin: 0;">标准符合性分析过程中出现异常，请检查PDF文件内容或稍后重试。</p>
            <details style="margin-top: 15px;">
                <summary style="color: #856404; cursor: pointer;">查看详细错误信息</summary>
                <pre style="background: #856404; color: white; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">{error_msg}</pre>
            </details>
        </div>
        """

def create_pdf_analysis_interface():
    """创建PDF分析界面"""
    print("[界面] 开始创建Gradio界面...", flush=True)
    
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
                # 提取的报告信息 - 使用HTML组件支持富文本显示
                report_info = gr.HTML(
                    label="提取的报告信息",
                    value="<p style='color: #666; font-style: italic;'>PDF分析结果将显示在这里...</p>",
                    show_label=True
                )
        
        # 标准符合性分析区域
        with gr.Row():
            with gr.Column():
                compliance_btn = gr.Button("分析标准符合性", variant="secondary")
                
                compliance_result = gr.HTML(
                    label="标准符合性分析结果",
                    value="<p style='color: #666; font-style: italic;'>标准符合性分析结果将显示在这里...</p>",
                    show_label=True
                )
        
        print("[界面] 界面组件创建完成，开始绑定事件...", flush=True)
        
        # 事件绑定 - 恢复到原始PDF处理函数
        analyze_btn.click(
            fn=process_pdf_file,
            inputs=[pdf_file],
            outputs=[status_text, report_info]
        )
        
        print("[界面] 事件绑定完成", flush=True)
        
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
    
    print("[界面] Gradio界面创建完成", flush=True)
    return interface

if __name__ == "__main__":
    print("[后台] ========== 启动PDF分析系统 ===========", flush=True)
    print(f"[后台] 工作目录: {WORKING_DIR}", flush=True)
    print(f"[后台] API地址: {BASE_URL}", flush=True)
    print(f"[后台] 视觉模型: {VL_MODEL}", flush=True)
    print(f"[后台] 嵌入模型: {EMBEDDING_MODEL}", flush=True)
    
    # 检查Gradio版本
    import gradio
    print(f"[后台] Gradio版本: {gradio.__version__}", flush=True)
    
    # 创建界面
    print("[后台] 正在创建Gradio界面...", flush=True)
    demo = create_pdf_analysis_interface()
    
    # 启动应用
    print("[后台] 正在启动应用服务器...", flush=True)
    print("[后台] 服务器地址: 127.0.0.1:10086", flush=True)
    print("[后台] ========== 系统启动完成 ===========", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=10086,
        share=True,  # 使用share=True来解决localhost访问问题
        debug=True,
        show_error=True,
        show_api=False
    )