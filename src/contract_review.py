import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table


class ContractReviewResult(BaseModel):
    rule_id:str = Field(
        description="审查规则编号"
    ) 
    status:Literal["符合要求","存在问题","无法判断"]=Field(
        description="依据合同文本作出的审查判断"
    )
    evidence:str = Field(
        description="合同原文证据"
    )
    issue:str = Field(
        description="发现的问题及原因"
    )
    suggestion:str = Field(
        description="合同修改建议"
    )

def load_contract_docx(file_path:str|Path)->str:
    """读取DOCX合同，按原始顺序提取段落和表格"""
    doc = Document(file_path)
    contract_blocks = []
    for block in doc.iter_inner_content():
        if isinstance(block,Paragraph):
            text = block.text.strip()
            if text:
                contract_blocks.append(text)
        elif isinstance(block,Table):
            block_text = []
            for row in block.rows:
                row_content = []
                for cell in row.cells:
                    row_content.append(cell.text.strip())
                block_text.append('|'.join(row_content))
            contract_blocks.append('表格内容\n'+'\n'.join(block_text))
    return '\n\n'.join(contract_blocks)

def load_contract_rules(rule_path:str) -> list[dict]:
    with open(rule_path,'r',encoding='utf-8') as f:
        rule_content = json.load(f)
    return rule_content

def review_contract_rule(contract_text:str,
                         rule:dict,
                         structured_llm
                         )->ContractReviewResult:

    prompts = f'''
    你是一名工业污水处理合同审查助手。
    请严格按照合同文本和规则进行审查。
    【审查规则】
    规则编号：{rule['rule_id']}
    规则名称：{rule['title']}
    审查问题：{rule['review_question']}
    预期要求：{rule['expected_requirement']}
    参考建议：{rule['suggestion']}
    【合同原文】
    {contract_text}
    【审查要求】
    1.只能根据合同文本进行判断，不得编造条款。
    2.如果合同已经满足要求，status返回“符合要求”
    3.如果合同存在缺失、表述不清或者不满足审查要求时，status返回“存在问题”
    4.如果现有文本不足以作出判断，status返回“无法判断”
    5.evidence必须直接复制合同原文的连续文字
    6.符合要求时，issue返回空字符串
    7.suggestion应给出有操作性的建议
    8.不得自行编造价格、超标倍数及公司制度
    9.涉及新增商务条件时，注明需双方进行确认
    10.rule_id必须与规则编号保持一致
                '''

    result = structured_llm.invoke(prompts)
    if isinstance(result,dict) and 'parsed' in result:
        if result['parsing_error'] is not None:
            raise ValueError(f'结构化解析失败原因为{result['parsing_error']}')
        result = result['parsed']
    if isinstance(result,dict):
        result = ContractReviewResult.model_validate(result)
    if result.rule_id != rule['rule_id']:
        raise ValueError('规则编号错误')
    return result
def review_contract(
    contract_text: str,
    rules: list[dict],
    structured_llm
) -> list[ContractReviewResult]:

    results = []

    for rule in rules:
        result = review_contract_rule(
            contract_text,
            rule,
            structured_llm
        )
        results.append(result)

    return results

def export_contract_report(
    results: list[ContractReviewResult],
    output_path: str | Path,
    contract_name: str
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    doc.add_heading("合同审查报告", level=1)
    doc.add_paragraph(f"合同名称：{contract_name}")
    sum_question = sum(1 for result in results if result.status == '存在问题')
    sum_ok = sum(1 for result in results if result.status == '符合要求')
    sum_unknown = sum(1 for result in results if result.status == '无法判断')
    doc.add_paragraph(f'存在问题的条款有{sum_question}项')
    doc.add_paragraph(f'无法判断需要补充的条款有{sum_unknown}项')
    doc.add_paragraph(f'不需要有修改的有{sum_ok}项')

    for result in results:
        doc.add_heading(result.rule_id, level=2)
        doc.add_heading(result.status, level=3)
        doc.add_paragraph(f'原文为：{result.evidence}')
        doc.add_paragraph(f'发现的问题为：{result.issue}')
        doc.add_paragraph(f'建议建议为：{result.suggestion}')
    doc.save(output_path)
    return output_path

def run_contract_review(
    contract_path: str | Path,
    rules_path: str | Path,
    output_path: str | Path,
    structured_llm,
    contract_name: str
) -> Path:
    contract_text = load_contract_docx(contract_path)
    rules = load_contract_rules(rules_path)
    results = review_contract(contract_text, rules, structured_llm)
    report_path = export_contract_report(results, output_path, contract_name)
    return report_path
    # 第一步：读取合同文本

    # 第二步：加载审查规则

    # 第三步：执行 review_contract()

    # 第四步：执行 export_contract_report()

    # 第五步：返回报告路径
