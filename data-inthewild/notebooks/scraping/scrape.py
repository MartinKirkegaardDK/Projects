from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, NoSuchWindowException,WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import sys
sys.path.append("../../src")
import scraping as f # Custom functions file containing all of the functions needed for this notebook

link = "https://tasty.co/tag/dinner" 

def driver_setup():
    driver_chrome = None
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver_chrome = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = options)
    driver_chrome.set_page_load_timeout(500)
    return driver_chrome

def driver_code(driver):
    with open('../../data/interim/scraping/has_scraped.json') as json_file:
        links = json.load(json_file)

    d = dict()
    for link, has_scaped in links.items():
        if has_scaped == True:
            continue
        print(f'Processing recipe: {link}')
        driver.get(link)

        time.sleep(5)
        if f.page_exists(driver):
            print("page doesn't exist")
            time.sleep(5)
            continue

        try:
            cookie = driver.find_element(By.ID,'onetrust-accept-btn-handler')
            cookie.click()
        except NoSuchElementException:
            pass

        print('\tGetting ingredients...')
        ingredients = [i.text for i in driver.find_elements(By.CLASS_NAME,'ingredient')]
        print('\tGetting nutrition...')
        nutrition = f.grab_nutrition(driver)
        print('\tGetting preparation...')
        preparation = f.grab_preparation(driver)
        print('\tGetting date...')
        date = f.grab_date(driver)
        print('\tGetting rating...')
        rating = f.grab_recipe_rating(driver)
        print('\tGetting tags...')
        try:
            tags = f.get_tags(driver)
        except NoSuchElementException:
            pass
        print('\tGetting number of comments...')

        try: 
            number_of_comments = driver.find_element(By.CLASS_NAME,'tips-count-heading').text
        except NoSuchElementException:
            number_of_comments = "0 TIPS"
        print('\tGetting comments...')
        if number_of_comments != "0 TIPS":
            comments = f.grab_comments(driver)
        else:
            comments = []

        d = {'ingredients': ingredients,
                                'nutrition': nutrition,
                                'preparation': preparation,
                                'date': date,
                                'rating': rating,
                                'tags': tags,
                                'number_of_comments': number_of_comments,
                                'comments': comments}
        
        f.save_and_remove_from_queue(link, d)
    time.sleep(5)
    return 'Done'


while True:
    driver_setup()
    print('Driver setup complete')
    f.get_urls(driver_setup(),link)
    print('URLs are scraped')
    f.create_link_queue('../../data/interim/scraping/links.txt')
    print('Created Link Queue')
    f.get_images(driver_setup())
    print('Images are scraped')
    driver_code(driver_setup())
    if driver_code(driver_setup()) == 'Done':
        driver_setup().close()
        print('Scraping Complete')