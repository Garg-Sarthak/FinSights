from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent


from .tools import analysis_tool, analyse_text, compare_texts
import json
import os
import sys

from llm_functions.analysis import summarise_sections, analyze_section_trends, compare_sections, get_sentiment
import concurrent.futures
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


if "GOOGLE_API_KEY" not in os.environ:
    gemini_api_key = os.getenv("gemini_api_key_3")
    try:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    except Exception as e:
        print("some issue in key")


sentiment_tool = StructuredTool.from_function(get_sentiment,name='sentiment_tool')
summary_tool = StructuredTool.from_function(summarise_sections,name='summary_tool',description="This is your go to tool, when you have to do a cross-company or cross-section or similar comparison. Use this tool to get a detailed summary of a SPECIFIC FINANCIAL SECTION for a SINGLE company in a SINGLE year.  This is the primary way to retrieve raw information before further analysis. Only allowed sections are : ['business_overview', 'management_discussion', 'risk_factors', 'financial_statements', 'legal_proceedings', 'competitive_landscape', 'macroeconomic_outlook', 'guidance_and_forecast', 'strategic_initiatives', 'capital_allocation', 'segment_reporting', 'revenue_drivers', 'cost_and_margin_analysis', 'shareholder_updates', 'regulatory_updates', 'technology_and_innovation']Any other section will result in error, even a wrong spelling.")
tools = [analysis_tool,sentiment_tool,summary_tool,compare_texts]
# tools = [sentiment_tool,summary_tool]
available_tools = {
    "analysis_tool" : analysis_tool,
    "summary_tool" : summary_tool,
    "sentiment_tool" : sentiment_tool,
    "compare_texts" : compare_texts
}


def agent(query):
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-pro",
        temperature = 0,
    )
    system_prompt = """
    You are "FinSight" an expert financial analyst assistant.
    Current year is 2025, so you have data till 2024/2025.
    IF CURRENT YEAR IS NOT GIVEN, ASSUME 2024. 
    Your primary goal is to use your available tools
    tool1 : summarise
    tool2 : compare texts
    tool3 : analysis_tool (this tool can only be used if multiple (more than 2 years) are given for a single company)
    tool4 : get sentiment (you can pass text to this, like what you get from summarise tool or compare texts tool)
    to answer user questions about financial documents, or finance in general.
    You must follow these rules strictly:
    for the term "section" here is a list of exhaustive sections : 
    ['business_overview', 'management_discussion', 'risk_factors', 'financial_statements', 'legal_proceedings', 'competitive_landscape', 'macroeconomic_outlook', 'guidance_and_forecast', 'strategic_initiatives', 'capital_allocation', 'segment_reporting', 'revenue_drivers', 'cost_and_margin_analysis', 'shareholder_updates', 'regulatory_updates', 'technology_and_innovation']
    1. First call tools, then respond, don't assume that you don't have information,
    If user doesn't mention year, assume latest year - 2024, and write since you didn't mention year I am assuming 2024.
    first try calling a tool, in case result is insufficient, only then say that data is insufficient, even in that case, make out what you can from parial data.
    2. If  a user doesn't clearly mentions a section, assume the closest section(s) from the exhaustive list yourself. 
    You may need to do a multi section analysis, requiring multiple steps, you can't expect user to tell everything, be smart.
    1. Analyze, Don't Assume: You MUST use your tools to find information. Do not answer from your own general knowledge.
    3. Admit When You Don't Know: If you cannot find the answer using your tools, you MUST state that the information is not available.
    4. Stay On Topic: If the user asks a non-financial question, you MUST politely refuse.
    5. Synthesize, Don't Just Report: When you have the final results from your tools, synthesize the information into a clear, concise , but extensive answer.
    """
    # 6. Currently you have tools for : comparing/analysing sections of a company over multiple years, summarise for a given section, compare a list of texts, get sentiment for some pieve of text.
    




    llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=True)
    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=query))
    
    while True:
        print("\n--- AGENT STEP ---")
        ai_response = llm_with_tools.invoke(messages)

        if not ai_response.tool_calls:

            print("\n--- AGENT FINISHED ---")
            return ai_response.content
        
        print(f"Agent will call - {[tool_call['name'] for tool_call in ai_response.tool_calls]}")
        messages.append(ai_response)
        # print(ai_plan.tool_calls)

        # for tool_call in ai_response.tool_calls:
        #     tool_function = available_tools.get(tool_call['name'])
        #     if tool_function:
        #         print(f"--- Agent executing tool: {tool_call['name']} ---")
        #         tool_output = tool_function.invoke(tool_call["args"])
        #         messages.append(ToolMessage(
        #                 content=json.dumps(tool_output),
        #                 tool_call_id=tool_call["id"],
        #             )
        #         )
        #     else:
        #       print(f"Agent tried to call unknown tool '{tool_call['name']}'")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_call = {
            executor.submit(
                available_tools[tc["name"]].invoke, 
                tc["args"]
                ): tc
                for tc in ai_response.tool_calls
                if tc["name"] in available_tools
            }

            for future in concurrent.futures.as_completed(future_to_call):
                tc = future_to_call[future]
                result = future.result()
                print(f"--- Tool {tc['name']} finished ---")
                messages.append(
                    ToolMessage(content=json.dumps(result),tool_call_id=tc["id"])
            )

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query : ")
        try:
            # ai_msg = llm_with_tools.invoke("infosys 2022 and 2024 revenue")
            res = agent(user_query)
            print(res)
        except Exception as e:
            print(e)