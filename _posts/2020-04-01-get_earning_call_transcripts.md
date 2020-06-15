---
title: Getting Earnings Call Transcripts
date: 2020-04-01
#tags: [Python, Earnings Call, NLP, Preprocess]
header:
  image: "/images/boardroom.jpg"
mathjax: "true"
---

# Part 1: Collecting Earnings Call Transcripts

After browsing the web for a few hours, I wasn't able to find any prepared datasets for earnings call transcripts but I did find some ways to collect our own data.  We will first look into scraping Seeking Alpha for the latest earnings call transcripts and also look at downloading PDFs of earnings call transcripts from the Investor Relations website of companies and parse the transcript from the PDF into CSV format. 


## Part 1a: Scraping Seeking Alpha for Earnings Transcripts

I implemented the git repo [here](https://github.com/stefan-jansen/machine-learning-for-trading/tree/master/03_alternative_data/02_earnings_calls) to gather earnings transcripts from Seeking Alpha into csv files.

Here's an example of what the scraped csv looks like (transcript for Easterly Government Properties Inc. (DEA) - Feb 25, 2020):

[![](/assets/images/get_earning_call_transcripts/raw_csv.JPG)](/assets/images/get_earning_call_transcripts/raw_csv.JPG)



The main issue with gathering earnings transcripts using this method is that we need a Seeking Alpha subscription to view earnings calls that are older than ~ 10 months.

For example, if I try to gather all the transcripts for a particular ticker, I would hit a page asking me to subscribe:

[![](/assets/images/get_earning_call_transcripts/subscribe.JPG)](/assets/images/get_earning_call_transcripts/subscribe.JPG)


Trying to avoid any sort of fees at this point since this is more for exploratory data analysis, but if we find interesting insights afterwards a subscription would be reasonable. 



## Part 1b: Converting Earnings Transcript PDFs to CSVs

This leads us to the second method of manually downloading the earnings transcripts and parsing through to convert it into a csv format similar to the one from scraping Seeking Alpha. 

For now, I've just randomly picked some tickers to download all the historic earnings transcript PDFs. 

We'll use the pdfplumber package to read in these PDFs in Python.

```cmd
pip install pdfplumber
```

Once you've downloaded that PDFs, you'll notice that the main providers of these earnings transcripts are Thomson Reuters, Factset and Bloomberg although some companies have their own format.

We can write some standard functions to parse these PDFs given that ones from the same providers are usually in the same format.

Here's a function I wrote to parse the Thomson Reuters format:

```python
def thom_reut_convert(ticker):
    path = os.path.join(paths['trans_hist'], ticker)
    pdf_list = [x for x in os.listdir(path) if x[-4:] == '.pdf']
    for f in pdf_list:
        found_start = False
        out = ''
        with pdfplumber.open(os.path.join(path, f)) as pdf:
            for i,v in enumerate(pdf.pages):
                curr_page = pdf.pages[i]
                txt = curr_page.extract_text()

                # parse date, quarter, year of call 
                if i == 0:
                    if txt.find(ticker + ' -') != -1:
                        dt_idx = txt.find(ticker + ' -')
                        qt_txt = txt[dt_idx:].split('\n')
                        qt_txt = [x for x in qt_txt if x != ' '][0].strip(ticker + ' -')  # remove ' '
                    else:
                        qt_txt = [x for x in txt.split('\n') if 'Earnings' in x][0]
                    quarter = qt_txt.split()[0]   # not using right now
                    year = qt_txt.split()[1]     # not using right now
                    date_idx = txt.find('EVENT DATE/TIME: ') + len('EVENT DATE/TIME: ')
                    date_idx_end = date_idx + 50  # static charater count for length of date
                    date = parse(txt[date_idx:date_idx_end].split('\n')[0]).strftime('%m-%d-%y')
                    comp_name = qt_txt.split(' Earnings Call')[0][qt_txt.split(' Earnings Call')[0].find(year)+ len(year) + 1:]

                elif i == 1:   # parse names of call participants, separated into corporate participants and conference participants
                    corp_idx = txt.find('CORPORATE PARTICIPANTS')+len('CORPORATE PARTICIPANTS\n')
                    corp_idx_end = txt.find('\nCONFERENCE CALL PARTICIPANTS')
                    conf_idx = txt.find('CONFERENCE CALL PARTICIPANTS') + len('CONFERENCE CALL PARTICIPANTS\n')
                    conf_idx_end = txt.find('PRESENTATION')
                    corp_txt = txt[corp_idx:corp_idx_end].split('\n')
                    corp_name = [x.split(comp_name.split()[0])[0] for x in corp_txt]
                    corp_name = [x.strip() for x in corp_name if x.strip() != '']
                    corp_speaker1 = corp_name     # for cases where first name, initial. last name
                    corp_speaker2 = [x.split()[0] + ' ' + x.split()[-1] for x in corp_name]

                    conf_txt = txt[conf_idx:conf_idx_end].split('\n')
                    conf_speaker1 = [x.split()[0] + ' ' + x.split()[1] for x in conf_txt if len(x.split()) >= 2]
                    conf_speaker2 = [x.split()[0] + ' ' + x.split()[2] for x in conf_txt if len(x.split()) >= 3]    # accounts for speakers with middle name
                    conf_speaker3 = [x.split()[0] + ' ' + x.split()[1] + ' ' + x.split()[2] for x in conf_txt if len(x.split()) >=3]    # accounts for speakers with middle name

                    speaker_list = corp_speaker1 + corp_speaker2 + conf_speaker1 + conf_speaker2 + conf_speaker3  

                    columns = ['corp_particip', 'conf_particip']
                    df_speaker = pd.DataFrame(columns=columns)
                    corp_list = corp_speaker1 + corp_speaker2
                    conf_list = conf_speaker1 + conf_speaker2 + conf_speaker3
                    df_speaker['corp_particip'] = corp_list + (max(len(corp_list), len(conf_list)) -len(corp_list)) * ['']
                    df_speaker['conf_particip'] = conf_list + (max(len(corp_list), len(conf_list)) -len(conf_list)) * ['']
                    df_speaker.to_csv(os.path.join(path, ticker + '_' + date + '_speakers.csv'), index=False)

                # remove footer
                if i!=0 and i !=len(pdf.pages)-1:  # remove the page number and footer (should be on all pages except first page)
                    #footer_idx = curr_page.extract_text().find('\nTHOMSON REUTERS STREETEVENTS')
                    pg_num_idx = txt.find('\n' + str(i + 1))
                    #txt = curr_page.extract_text()[:footer_idx].split('\n')
                    txt = txt[:pg_num_idx]

                #remove disclaimer
                elif i ==len(pdf.pages)-1:  # if last page remove disclaimer
                    dis_idx = txt.find('DISCLAIMER')
                    txt = txt[:dis_idx]


                # write to file
                if found_start == True:
                    out = out+txt
                    #with open(os.path.join(path, ticker + '_' + date + '.txt'), 'a', encoding="utf-8") as myfile:
                        #myfile.write(txt)

                # found beginning of presentation, start writing to file (commented out part writes it to a text file)
                elif txt.find('PRESENTATION') != -1 and found_start ==False:
                    found_start = True
                    start_idx = txt.find('PRESENTATION')+len('PRESENTATION\n')
                    out = out+txt[start_idx:]
                    #with open(os.path.join(path, ticker + '_' + date + '.txt'), 'a', encoding="utf-8") as myfile:
                        #myfile.write(txt[start_idx:])

        # put it into same csv format as transcripts from Seeking Alpha
        df = pd.DataFrame(columns=columns)
        speaker = ''
        content = ''
        qna_start_j = 1000  # initialize
        j = 0
        for line in out.split('\n'):

            if line.find('QUESTIONS AND ANSWERS') != -1:
                qna_start_j = j + 2  # line that qna starts (+2 is to ignore current line and operator line)
            if len(line.split()) == 1:
                find_speaker = [x for x in ['Operator'] if (x in line)]
            else:
                find_speaker = [x for x in speaker_list if (x + ' -' in line)]

            if find_speaker == []:
                content = content + ' ' + line
            else:
                if speaker != '':
                    if j < qna_start_j:
                        df.loc[j] = [speaker, 0, content]
                    else:
                        df.loc[j] = [speaker, 1, content]
                    content = ''
                    j+=1
                speaker = find_speaker[0]

        df.to_csv(os.path.join(path, ticker + '_' + date + '.csv'), index=False)
        print('converted: ' + os.path.join(path, ticker + '_' + date + '.csv'))
```

# Part 2: Preprocessing Earnings Transcripts

Now that the raw transcripts are saved down, I need to do a bit of preprocessing to clean up the data.  In each quarterly earnings call starts off with the company's CEO/management team discussing their previous quarter's progress and future outlook followed by the Q&A session from various external analysts.  For this exercise, I will remove the mamangement discussion section and only analyze the sentiment of this Q&A session since it tends to provide more insight into areas the seasoned analysts focus on. 

While gathering the raw transcripts into csvs, the PDF to csv conversion functions and the Seeking Alpha scraper from the git repo was able to gather a list of names of the corporate (management) and conference (analysts) speakers for each earnings transcript.  I will use this list to further filter the transcript to only include the responses of management to the analysts questions.

There is a lot of greetings and common filler words like "Hi", "Thanks", "Bye" in the transcripts so I've saved down a list of filler words to look for and remove from the transcript if that line contains less than 10 words. Here's what's in that list right now:

[![](/assets/images/get_earning_call_transcripts/filler_words.JPG)](/assets/images/get_earning_call_transcripts/filler_words.JPG)


Here's the helper function to remove filler words that we'll use in our preprocess wrapper function:

```python
def fillers_to_na(text_list):
    # drop anything that contains a filler word and has len less than 10
    fillers_df = pd.read_csv(os.path.join(paths['trans'], 'filler_words.csv'))
    fillers = fillers_df['filler_words'].tolist()
    if any(filler in text_list for filler in fillers) and len(text_list)<10:
        text_list = np.nan

    # drop anything with 3 words or less
    elif len(text_list)<=3:
        text_list = np.nan

    return text_list
```


Similar to how we preprocessed the tweets, I copied the previous PreProcessTweets class and made slight adjustments to preprocess the transcript text:


```python
class PreProcessTexts:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTexts(self, list_of_texts):
        processedTexts = []
        featureVector = []
        for text in list_of_texts:
            processedText = text[0], text[1], self._processText(text[2])  
            processedTexts.append(processedText)
            featureVector.append(processedText[2]) 
        return processedTexts, featureVector

    def processTextsAsStr(self, list_of_texts):
        processedTexts = []
        featureVector = []
        for text in list_of_texts:
            processedText = text[0], text[1], self._processText(text[2])
            processedText_as_str = ''
            for word in processedText[2]:
                if processedText_as_str == '':
                    processedText_as_str = word
                else:
                    processedText_as_str = processedText_as_str + ' ' + word
            processedText_as_str_full = text[0], text[1], processedText_as_str
            processedTexts.append(processedText_as_str_full)
            featureVector.append(processedText[2])
        return processedTexts, featureVector


    def _processText(self, text):
        text = text.lower()  # convert text to lower-case
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)  # remove URLs
        text = re.sub('@[^\s]+', 'AT_USER', text)  # remove usernames
        text = re.sub(r'#([^\s]+)', r'\1', text)  # remove the # in #hashtag
        text = word_tokenize(text)  # remove repeated characters (helloooooooo into hello)
        text = [word for word in text if re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word) is not None]  # remove words that doesn't start with alphabet
        return [word.strip('\'"') for word in text if word not in self._stopwords and len(word) > 2]
```


Taking all of the above into consideration, I wrote this function to go through the folders that contain the "content.csv" we obtained from the Seeking Alpha scraper.  It will create a new "content_prep.csv" in each folder that contain into 2 new columns than "content.csv" which are the preprocessed columns. (One column has the preprocessed text tokenized and another column has the preprocessed text as a string):


```python
def create_preprocessed_transcript():
    t0 = time.time()
    folder_list = os.listdir(paths['trans_parsed'])
    for folder in folder_list:
        print(folder)
        path = os.path.join(paths['trans_parsed'],folder)
        df = pd.read_csv(os.path.join(path, 'content.csv'))
        df.columns = ['speaker', 'qna', 'content']   # to make the View as DataFrame work
        df = df.dropna()   # drop na

        # parse the company participant names
        # row name for company participants can change: (ie: 'Company Participants', 'Company Representatives')
        if df['speaker'].str.contains('Corporate').any():
            find_company_particip_string = df['speaker'][df['speaker'].str.contains('Corporate')].iloc[0]
            comp_particip = df[df['speaker'] == find_company_particip_string]['content'].iloc[0]
            comp_particip = [y[0] for y in [x.split(' - ') for x in comp_particip.split('\n')]]  # parse the company participant names
        elif df['speaker'].str.contains('Company').any():
            find_company_particip_string = df['speaker'][df['speaker'].str.contains('Company')].iloc[0]
            comp_particip = df[df['speaker'] == find_company_particip_string]['content'].iloc[0]
            comp_particip = [y[0] for y in [x.split(' - ') for x in comp_particip.split('\n')]]  # parse the company participant names
        else:
            particip_df = pd.read_csv(os.path.join(path, 'participants.csv'))
            name_list = particip_df[particip_df['type']=='Executives']['name'].tolist()
            comp_particip = [x.split(' - ')[0] for x in name_list]   # parse the company participant names

        if df[df['qna'] == 1].empty:  # if everything in qna column is 0, don't filter for qna==1  (means qna column is mis-labelled）
            filtered_df = df[(df['speaker'].isin(comp_particip))].reset_index(drop=True)
        else:
            filtered_df = df[(df['qna'] == 1) & (df['speaker'].isin(comp_particip))].reset_index(drop=True)

        # convert it to a list of lists to feed it to PreProcessTweets function
        filtered_df_list = filtered_df.values.tolist()

        # create a new csv that contains the filtered df with additional content_preprocessed column
        text_processor = utilities.PreProcessTexts()
        preprocessed_training_set, feature_list = text_processor.processTexts(filtered_df_list)
        preprocessed_training_set_as_str, feature_list_as_str = text_processor.processTextsAsStr(filtered_df_list)
        feature_list = list(set(list(chain.from_iterable(feature_list))))  # flatten nested list and remove duplicates

        # save down feature_list (currently not using)
        with open(os.path.join(path,'feature_list'), 'wb') as fp:
            pickle.dump(feature_list, fp)

        # this is used for normal vectorizer
        preprocessed_df = pd.DataFrame(preprocessed_training_set, columns=df.columns)
        filtered_df['preprocessed'] = preprocessed_df['content']

        # needs to be in this format to feed into nltk ngram
        preprocessed_as_str_df = pd.DataFrame(preprocessed_training_set_as_str, columns=df.columns)
        filtered_df['preprocessed_as_str'] = preprocessed_as_str_df['content']

        # drop filler rows
        filtered_df['preprocessed'] = filtered_df['preprocessed'].apply(fillers_to_na)
        filtered_df = filtered_df.dropna()

        filtered_df.to_csv(os.path.join(path, 'content_prep.csv'), index=False)

    t1 = time.time()
    print(t1-t0)
```

Here's an example of what the preprocessed csv looks like (transcript for Signet Jewelers Ltd. (SIG) - March 26, 2020):

[![](/assets/images/get_earning_call_transcripts/preprocessed_csv.JPG)](/assets/images/get_earning_call_transcripts/preprocessed_csv.JPG)


Now we will aggregate all of these individual quarterly earnings call csvs into one giant csv that will be fed into our model to be predicted. 

Here's the function to combine the preprocessed csvs:

```python
def create_combined_transcript():
    # combine the transcript csvs into 1 df
    combined_df = pd.DataFrame(columns= ['speaker', 'qna', 'content', 'preprocessed', 'preprocessed_as_str', 'sentiment', 'folder'])

    # save down the folders that have an error and move on to the next folder
    errors_df = pd.DataFrame(columns=['folder'])
    j=0

    folder_list = os.listdir(paths['trans_parsed'])
    for folder in folder_list:
        print('combining transcripts csv: ' + folder)
        try:
            path = os.path.join(paths['trans_parsed'],folder)
            df = pd.read_csv(os.path.join(path, 'content_prep.csv'))
            #df = pd.read_csv(os.path.join(path, 'content.csv'))
            df.columns = ['speaker', 'qna', 'content', 'preprocessed', 'preprocessed_as_str']
            df['sentiment'] = 2
            df['folder'] = folder

            combined_df = combined_df.append(df)
        except:
            errors_df['folder'].loc[j] = folder
            j+=1

    # re-name the column to 'review' for the column you want to feed to model (tokenizer looks for 'review' column)
    combined_df = combined_df.rename(columns={'preprocessed_as_str':'review'})

    errors_df.to_csv(os.path.join(paths['trans'], 'errors_combined_trans.csv'), index=False)
    combined_df.to_csv(os.path.join(paths['trans'], 'combined_trans.csv'), index=False)
    print('combined transcripts csv saved: ' + str(os.path.join(paths['trans'], 'combined_trans.csv')))
```




