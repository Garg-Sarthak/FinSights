from google import genai
from google.genai import types
import dotenv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

from ingestor.assign_sections import get_sections
from search.intelli_search import get_section_chunks

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.intelli_search import get_section_chunks
from typing import List,Dict,Any

import concurrent.futures

def get_client():
    # api_key = dotenv.get_key("../.env","gemini_api_key")
    dotenv.load_dotenv()
    api_key = os.getenv("gemini_api_key")
    client = genai.Client(api_key=api_key)
    # print(api_key)
    return client

def summarise_section_main(chunk,company:str,year:Any,section:str) -> str:
    try:
        year = (int)(year)
        # chunk = chunks[year]
        client = get_client()
        SAFE_CHAR_LIMIT = 300_000/2
        SAFE_TOKEN_LIMIT = 1_000_000/1.5
        
        batches = []
        current_batch = ""
        for chunk_text in chunk:
            if (len(chunk_text) + len(current_batch) > SAFE_CHAR_LIMIT):
                batches.append(current_batch)
                current_batch = chunk_text
            else:
                current_batch += "\n\n" + chunk_text
        batches.append(current_batch)

        intermediate_summaries = []
        print(f"Mapping {len(batches)} batch(es) for {company} {year}...")

        # chunk_text = "\n".join(chunk)
        for i,batch_text in enumerate(batches):
            map_prompt = f"Summarize the key points in the following text excerpt focus mainly on the topic {section} (this process is under a batching mechanism which will be used lastly for analysis, so try to not miss out on any information, specially if it is related to numbers). DON'T OMMIT OUT ANY FACT OF FINANCIAL INTEREST, SPECIFICALLY DATA WHICH IS SUPPOSEDLY UNDER FINANCIAL STATEMENTS: \n\n{batch_text}"
            map_response = client.models.generate_content(model="gemini-2.5-flash",contents=map_prompt,
                                                            config=types.GenerateContentConfig(max_output_tokens=(int)(SAFE_TOKEN_LIMIT)))
            intermediate_summaries.append(map_response.text)
            print(f"batch {i+1} done for {company} {year}...")
        

        print(f"successfully mapped {len(batches)} for {company}_{section}_{year}")
        combined_summary_text = "\n".join(intermediate_summaries)
        system_prompt = f"You are a SENIOR FINANCIAL ANALYST. Based on the following text ONLY under '{section.capitalize()}' disclosed by {company.capitalize()} in their {year} annual report. You are supposed to create an EXTENSIVE but CONCISE summary for it. In case, the section is related to financial statements, try to be as extensive as possible, don't miss out on any facts of accounting or financial interest, otherwise you can be a bit more concise. If the RELEVANT content is not in text, say the data provided is insufficient, and move on. DON'T make ASSUMPTIONS. Provide it as a formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=combined_summary_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        print(f"successfully generated summary for {company}_{section}_{year}")
        return response.text or ""
    except Exception as e:
        print(f"Error summarizing {company} {year}: {e}")
        return ""

def summarise_sections(company:str,years:List[int],section):
    """
    Use this tool to get a detailed summary of a SPECIFIC FINANCIAL SECTION for a SINGLE company in a SINGLE year. 
    This is the primary way to retrieve raw information before further analysis.
    Only allowed sections are : ['business_overview', 'management_discussion', 'risk_factors', 'financial_statements', 'legal_proceedings', 'competitive_landscape', 'macroeconomic_outlook', 'guidance_and_forecast', 'strategic_initiatives', 'capital_allocation', 'segment_reporting', 'revenue_drivers', 'cost_and_margin_analysis', 'shareholder_updates', 'regulatory_updates', 'technology_and_innovation']
    Any other section will result in error, even a wrong spelling.
    """
    try:
        company = company.lower()    
        chunks = get_section_chunks(section=section,company=company,years=years)
        # if not all(chunks.values()):
        #     print("\nError: Could not retrieve document chunks. Please ensure data for all specified years has been ingested and there are enough.")
        #     return
        summaries = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_year = {
                executor.submit(summarise_section_main,chunks[year], company, year,section) : year for year in years
            }
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                result = future.result()
                summaries[year] = result
    except Exception as e:
        print(f"error while parallel summarising : {e}")
    return dict(sorted(summaries.items()))




def analyze_section_trends(summaries_by_year: Dict[int, str], company: str, section: str, years: List[int|str]) -> str:
    assert len(years) >= 2, f"Trend for less than 2 years can't be compared"
    years.sort()

    try:

        # year_1 = int(years[0])
        # year_2 = int(years[-1])
        # eff_years = [year_1,year_2]

        # summaries = {
        #     year_1 : summaries_by_year[year_1],
        #     year_2 : summaries_by_year[year_2]
        # }
        # summaries = str(summaries_by_year)
        comparison_text = ""
        for year in years:
            year = (int)(year)
            text = summaries_by_year[year]
            comparison_text = comparison_text + f"year_{year} : {text}"


        client = get_client()
        system_prompt = f"""
        You are a Senior Financial Analsyst. 
        You will be provided with some information/data for the company {company},
        under the section {section},
        for the years : {years}.
        identify and describe the key trends, evolving themes, 
        and the trajectory of sentiment over the period from {years[0]} to {years[-1]}.
        You are required to compare and analyse them and generate a bullet-point comparison. 
        In case, the section is related to financial statements or accounting information, your answer will be used to find information related to fundamental/financial analysis, so answer in accordance.
        Keep the result extensive, but concise, to the point.
        Don't assume any other unobvious data, only do what ever is possible from the provided data.
        If data is insufficient, you may tell so, but in no case make you own data.
        Provide formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading
        """

        response = client.models.generate_content(
            model = "gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents = comparison_text
        )
        return response.text or "no response"
    except Exception as e:
        print('error while comparing')
        print(e)

def compare_sections(summaries_by_year: Dict[int, str], company: str, section: str, years: List[int|str]) -> str:
    assert len(years) >= 2, f"Less than 2 years can't be compared"
    years.sort()

    try:

        year_1 = int(years[0])
        year_2 = int(years[-1])
        eff_years = [year_1,year_2]

        # summaries = {
        #     year_1 : summaries_by_year[year_1],
        #     year_2 : summaries_by_year[year_2]
        # }
        # summaries = str(summaries_by_year)
        comparison_text = f"""
        Summary for {year_1}:
        ---
        {summaries_by_year[year_1]}
        ---

        Summary for {year_2}:
        ---
        {summaries_by_year[year_2]}
        ---
        """


        client = get_client()
        system_prompt = f"""
        You are a Senior Financial Analsyst. 
        You will be provided with some information/data for the company {company},
        under the section {section},
        for the years : {years}.
        You are required to compare and analyse them and generate a bullet-point comparison. 
        In case, the section is related to financial statements or accounting information, your answer will be used to find information related to fundamental/financial analysis, so answer in accordance.
        Explicitly label changes as 'ADDED', 'REMOVED', or 'MODIFIED'. Focus only on significant differences.
        Keep the result extensive, but concise, to the point.
        Don't assume any other unobvious data, only do what ever is possible from the provided data.
        If data is insufficient, you may tell so, but in no case make you own data.
        Provide formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading.
        """

        response = client.models.generate_content(
            model = "gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents = comparison_text
        )
        return response.text or "no response"
    except Exception as e:
        print('error while comparing')
        print(e)
    
def get_sentiment(text: str) -> dict:
    """
    Use this tool when you need to find the sentiment (Positive, Negative, or Neutral) of a specific piece of text. The input must be the text itself, not a company or year.
    Provided a body of text, returns the sentiment for it as a dictionary : {sentiment,score}
    Uses FinBERT, to ensure that sentiment is in accordance with financial text / commentary
    """

    MODEL_NAME = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = softmax(outputs.logits[0].numpy())

        max_score_index = scores.argmax()
        sentiment_label = model.config.id2label[max_score_index]
        confidence_score = scores[max_score_index]

        return {
            "sentiment": sentiment_label.capitalize(),
            "score": float(confidence_score)
        }

    except Exception as e:
        print(f"Error ovccured during sentiment analysis: {e}")
        return {"sentiment": "Error", "score": 0.0}



# don't use this, use parallel version above
def summarise_section(chunks:Dict[int,List[str]],company:str,years:List[int],section):

    try:
        client = get_client()
        summaries = {}
        SAFE_CHAR_LIMIT = 300_000/2
        SAFE_TOKEN_LIMIT = 1_000_000/1.5

        for year in years:
            year = (int)(year)
            chunk = chunks[year]
            
            batches = []
            current_batch = ""
            for chunk_text in chunk:
                if (len(chunk_text) + len(current_batch) > SAFE_CHAR_LIMIT):
                    batches.append(current_batch)
                    current_batch = chunk_text
                else:
                    current_batch += "\n\n" + chunk_text
            batches.append(current_batch)

            intermediate_summaries = []
            print(f"Mapping {len(batches)} batch(es) for {company} {year}...")

            # chunk_text = "\n".join(chunk)
            for batch_text in batches:
                map_prompt = f"Summarize the key points in the following text excerpt focus mainly on the topic {section} (this process is under a batching mechanism which will be used lastly for analysis, so try to not miss out on any information, specially if it is related to numbers). DON'T OMMIT OUT ANY FACT OF FINANCIAL INTEREST, SPECIFICALLY DATA WHICH IS SUPPOSEDLY UNDER FINANCIAL STATEMENTS: \n\n{batch_text}"
                map_response = client.models.generate_content(model="gemini-2.5-flash",contents=map_prompt,
                                                              config=types.GenerateContentConfig(max_output_tokens=(int)(SAFE_TOKEN_LIMIT)))
                intermediate_summaries.append(map_response.text)
            

            combined_summary_text = "\n".join(intermediate_summaries)
            system_prompt = f"You are a SENIOR FINANCIAL ANALYST. Based on the following text ONLY under '{section.capitalize()}' disclosed by {company.capitalize()} in their {year} annual report. You are supposed to create an EXTENSIVE but CONCISE summary for it. In case, the section is related to financial statements, try to be as extensive as possible, don't miss out on any facts of accounting or financial interest, otherwise you can be a bit more concise. If the RELEVANT content is not in text, say the data provided is insufficient, and move on. DON'T make ASSUMPTIONS. Provide it as a formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=combined_summary_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            summaries[year] = response.text
        return summaries
    except Exception as e:
        print("error while summarising")
        print(e)
        return None


        

if __name__ == "__main__":
    scn = "strategic_initiatives"
    yrs = [2022,2023]
    res = get_section_chunks(section=scn,company="infosys",years=yrs)
    summ = summarise_sections(chunks=res,company="infosys",section=scn,years=yrs)
    # summ = analyze_section_trends(summ,"infosys","revenue_drivers",[2022,2024])
    print(summ)