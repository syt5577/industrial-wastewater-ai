from pydantic import Field,BaseModel
class PricingInput(BaseModel):
    water_quality: dict[str,float] = Field(
        description=(
            "工业污水水质指标及浓度。"
            "COD、化学需氧量 → COD_mg_L；"
            "氨氮、NH4-N → NH4_N_mg_L；"
            "总磷、TP → TP_mg_L；"
            "总氮、TN → TN_mg_L；"
            "BOD5、生化需氧量 → BOD5_mg_L；"
            "悬浮物、SS → SS_mg_L；"
            "只提取用户提供的指标，浓度默认单位为mg/L。")
            )

