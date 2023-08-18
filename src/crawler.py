from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json


def extract_page_content(url):
    # Initialize Selenium web driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    try:
        # Load web page
        driver.get(url)

        # Wait for the page to fully load
        wait = WebDriverWait(driver, 1)
        page_section_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'topic'))
        )

        # Initialize a BeautifulSoup object from the page section element's HTML content
        page_section_soup = BeautifulSoup(page_section_element.get_attribute('outerHTML'), 'html.parser')

        # Extract the title from the heading tag
        title_element = page_section_soup.find('h1')
        if title_element is None:
            print(f"Title not found for URL: {url}")
            title = "N/A"
        else:
            title = title_element.text.strip()

        # Extract the content from the paragraph tags
        content_elements = page_section_soup.find_all('p')
        content = "\n".join([p.text.strip() for p in content_elements])

        # Create a dictionary with the title and content of the page
        page_data = {
            "title": title,
            "content": content,
            "url": url
        }

        print(page_data)
        add_page(page_data)
        return page_data

    finally:
        # Close the web driver
        driver.quit()


def fetch_links_from_page(url):
    # Initialize Selenium web driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    try:
        # Load web page
        print(f"Visiting URL: {url}")
        driver.get(url)

        # Wait for the page to fully load
        time.sleep(1)

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse the page source and extract the links
        soup = BeautifulSoup(page_source, 'html.parser')

        # Find the <ul> element
        ul_element = soup.find('ul', style='display: block;')  # Find the <ul> element with the specified style

        # Initialize an empty list to store page data
        all_page_data = []

        # Extract the page content
        page_data = extract_page_content(url)
        if page_data['content']:
            all_page_data.append(page_data)

        if ul_element:
            # Extract the links from the <ul> element
            links = [f"https://docs.data.world/en/{link['href']}" for link in ul_element.find_all('a')]

            # Process each link and get page data
            for link in links:
                print(f"Visiting URL: {link}")

                # Recursively call the function for sub links
                sub_page_data = fetch_links_from_page(link)

                # Add the URL to each sub-page data entry
                for entry in sub_page_data:
                    entry['url'] = link

                # Extend the list with sub-page data
                all_page_data.extend(sub_page_data)

        return all_page_data

    finally:
        # Close the web driver
        driver.quit()


def scrape_doc_page(doc_url):
    # Recursively crawls through each main doc link and extracts the page content
    fetch_links_from_page(doc_url)


def add_page(page):
    # Read existing JSON data from the file
    existing_data = []
    try:
        with open('v5_scrape.json', 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        pass

    # Append the new JSON object to the existing data
    existing_data.append(page)

    # Write the updated data back to the file
    with open('v5_scrape.json', 'w') as json_file:
        json.dump(existing_data, json_file, indent=2)
