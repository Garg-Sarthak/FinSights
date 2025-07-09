from google import genai
from google.genai import types
import dotenv
import os


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.intelli_search import get_section_chunks

from typing import List,Dict,Any

def get_client():
    # api_key = dotenv.get_key("../.env","gemini_api_key")
    dotenv.load_dotenv()
    api_key = os.getenv("gemini_api_key")
    client = genai.Client(api_key=api_key)
    # print(api_key)
    return client

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
                map_prompt = f"Summarize the key points in the following text excerpt focus mainly on the topic {section} (this process is under a batching mechanism which will be used lastly for analysis, so try to not miss out on any information): \n\n{batch_text}"
                map_response = client.models.generate_content(model="gemini-2.5-flash",contents=map_prompt,
                                                              config=types.GenerateContentConfig(max_output_tokens=(int)(SAFE_TOKEN_LIMIT)))
                intermediate_summaries.append(map_response.text)
            

            combined_summary_text = "\n".join(intermediate_summaries)
            system_prompt = f"You are a SENIOR FINANCIAL ANALYST. Based on the following text ONLY under '{section.capitalize()}' disclosed by {company.capitalize()} in their {year} annual report. You are supposed to create an EXTENSIVE but CONCISE summary for it. If the RELEVANT content is not in text, say the data provided is insufficient, and move on. DON'T make ASSUMPTIONS. Provide it as a formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading"
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
        Explicitly label changes as 'ADDED', 'REMOVED', or 'MODIFIED'. Focus only on significant differences.
        Keep the result extensive, but concise, to the point.
        Don't assume any other unobvious data, only do what ever is possible from the provided data.
        If data is insufficient, you may tell so, but in no case make you own data.
        Provide formal output, don't tell I am AI, or I will do this and this, just come to the main point instantly, with a concise heading.
        """

        response = client.models.generate_content(
            model = "gemini-2.0-flash-lite",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents = comparison_text
        )
        return response.text or "no response"
    except Exception as e:
        print('error while comparing')
        print(e)
    

    




        

if __name__ == "__main__":
    res = get_section_chunks(section="financial_statements",company="infosys",years=[2024,2023,2222])
    summ = summarise_section(chunks=res,company="infosys",section="risk factors",years=[2023,2024])
    print(summ)