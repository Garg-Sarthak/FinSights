from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List,Dict,Any
import google.generativeai as genai
from google.genai import types
from google import genai
import os
import sys
from dotenv import load_dotenv
import concurrent.futures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_functions.analysis import summarise_sections, analyze_section_trends, compare_sections, get_sentiment
from ingestor.assign_sections import get_sections

def __init__():
    pass

load_dotenv()
gemini_key = os.getenv("gemini_api_key_3")

class AnalysisArgs(BaseModel):
    company: str = Field(description="The name of the company to analyze.")
    section: str = Field(description="The financial section to analyze. ")
    years: List[int] = Field(description="A list of two or more years to analyze.")

@tool(args_schema=AnalysisArgs)
def analysis_tool(company: str, section: str, years: List[int]) -> Dict:
    """
    Use this tool to perform a comprehensive financial comparison on a company's documents for a given section and a list of more than 1 years. 
    It will automatically perform a 2-year comparison if two years are provided, or a multi-year trend analysis if more than two years are provided.
    This tool ALSO returns the sentiment analysis for each year.
    This is primary tool when you have to perform a analysis for a SINGLE company for a single section over MULTIPLE years
    Any other section will result in error.
    """
    # Only allowed sections are : ['business_overview', 'management_discussion', 'risk_factors', 'financial_statements', 'legal_proceedings', 'competitive_landscape', 'macroeconomic_outlook', 'guidance_and_forecast', 'strategic_initiatives', 'capital_allocation', 'segment_reporting', 'revenue_drivers', 'cost_and_margin_analysis', 'shareholder_updates', 'regulatory_updates', 'technology_and_innovation']
    print(f"--- Agent Tool: Running analysis for {company}, {section}, {years} ---")
    try:
        summaries = summarise_sections(company,years,section)
        if not summaries:
            return {"error" : "failed to get summaries"}
        
        sentiment_results = {}
        for year in years:
            summary_text = summaries[year]
            sentiment_results[year] = get_sentiment(summary_text)
        
        report_text = ""
        analysis_type = ""

        if (len(years) == 2):
            analysis_type = "comparison"
            report_text = compare_sections(summaries,company=company,section=section,years=years)
        else:
            analysis_type = "multi year trend analysis"
            report_text = analyze_section_trends(summaries,company=company,section=section,years=years)

        return {
            "analysis_type" : analysis_type,
            "report" : report_text,
            "sentiment_results" : sentiment_results
        }

    except Exception as e:
        print(f"error while performing financial analysis")
        return {
            "analysis_type" : "",
            "report" : "",
            "sentiment_results" : 0
        }


class CompareArgs(BaseModel):
    texts : List[str] = Field(description="It contains the texts that needs to be compared")
    topic: str = Field(description="The topic of the comparison.")

class AnalyseArgs(BaseModel):
    texts : str = Field(description="It contains the texts that needs to be analysed")
    topic: str = Field(description="The topic of the comparison.")
    return_len : int = Field(description="Maximum length of output, keep it low to prevent api limits exceeding")

def analyse_text(text:str,topic:str,return_len:int) -> str:
    """
    This tool analyses a given text, and reduces it to a summary, while retaining information. 
    """
    return_len = min(200_000,return_len)
    client = genai.Client(api_key=gemini_key)

    response = client.models.generate_content(model="gemini-2.5-pro",contents=f"Given the {text}, you are supposed to analyse and summarise it, in context of the topic {topic}. \
                                    Keep it concise, but retain information relevant to the topic {topic}. Your reponse will later be used to make another decisions so. \
                                    Also don't tell I am an AI, I will do this or that, just give a concise heading - {topic}, and then your analysis\
                                    Also don't make information out of assumptions, nothing more than the provided text. ",config=types.GenerateContentConfig(max_output_tokens=return_len))
    return f"{topic} ----> {response.text}"


@tool(args_schema=CompareArgs)
def compare_texts(texts:List[str],topic) -> str:
    """
    Provided a list of texts, this tools compares it, in context of a given topic. It makes use of another tool analysis tool.
    """
    client = genai.Client(api_key=gemini_key)

    # genai.configure(api_key=gemini_key)

    len_texts = sum([len(text) for text in texts])
    total_tokens = 200_000
    final_text = ""

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for text in texts : 
            futures.append(executor.submit(analyse_text,text,topic,(int)(total_tokens*len(text)/len_texts)))
        i = 1
        for future in concurrent.futures.as_completed(futures):
            print(f"{i} summarised out of {len(texts)}")
            i += 1
            final_text += f"Another summary begins : {future.result()} Another summary ends."
    
    # llm = genai.GenerativeModel('gemini-2.5-pro',system_instruction=)
    response=client.models.generate_content(model="gemini-2.5-flash-lite",
                                   config=types.GenerateContentConfig(system_instruction="I will proivde you with a text containing multiple summaries, and a topic. You are supposed to do a comparitive analysis for them, like each summary is for some topic. Try to keep it extensive, but concise.  Also don't tell I am an AI, I will do this or that, just give a concise heading - {topic}, and then your analysis.Also don't make information out of assumptions, nothing more than the provided text. ",),
                                   contents=final_text)
    return response.text




if __name__ == "__main__":
    # res = analysis_tool.invoke({"company":"infosys","section":"strategic_initiatives","years":[2022,2023]})
    # print(res)

    # ana = analyse_text('"["this is text 1","this is text2"]"',"trial",10000)
    # print(ana)
    print(compare_texts.invoke(input={"texts":["this is text 1","this is text2"], "topic":"trial"}))