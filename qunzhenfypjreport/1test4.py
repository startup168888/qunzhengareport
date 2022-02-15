import pandas as pd
from docx import Document
import time
import sys

#import all the csv file, (rename to your file path)
df= pd.read_csv('C:/Users/qunzh/Documents/fypj/testing.csv')
co= pd.read_csv('C:/Users/qunzh/Documents/fypj/conversionrate.csv')
lo = pd.read_csv('C:/Users/qunzh/Documents/fypj/location1.csv')
sn = pd.read_csv('C:/Users/qunzh/Documents/fypj/socialnetwork.csv')
ts = pd.read_csv('C:/Users/qunzh/Documents/fypj/trafficsource.csv')
ur = pd.read_csv('C:/Users/qunzh/Documents/fypj/userretention.csv')
nr = pd.read_csv('C:/Users/qunzh/Documents/fypj/newreturn.csv')
br = pd.read_csv('C:/Users/qunzh/Documents/fypj/browser.csv')
#export to text file, open it to write all the content below
sys.stdout = open("test2.txt", "w")

print("Section 1 Page View and View Duration")

#Finding the highest page view data and print the result out
d=df['Pageviews'].max()
#Finding the lowest page view data and print the result out
ld=df['Pageviews'].min()
#find the highest view duration
hvd = df['Avg. Time on Page'].max()
#find the lowest view duration
lvd = df['Avg. Time on Page'].min()

#Use for loop to find the page title that match the highest page view and print the page title, pageview, bounce rate and and average time for the HIGHEST PAGE VIEW ONLY
#find highest page duration that matches with it
for ind in df.index:
    #check for the page title that has the highest page view and have the longest average time on page
    if df['Pageviews'][ind] == d and df['Avg. Time on Page'][ind] ==hvd:
         print(df['Page Title'][ind], 'has the most popular page view of ', df['Pageviews'][ind],'.',
               df['Page Title'][ind], 'has the \nhighest view duration of  Average Time on Page', round(df['Avg. Time on Page'][ind]//60),':',round(df['Avg. Time on Page'][ind]-(df['Avg. Time on Page'][ind]//60)*60), 'minutes.')
         print("This implies that more emphasise should be place on this page to attract the attention of the users \nby having a more interative user experience, better presentation of the website and most importantly \nhighlight the main products that you want to showcase."
                "Additional enhancement can be having reviews \nand testimonals, a scroll-to-top button to further attract the user attention in this website.")

#Use for loop to find the page title that match the lowest page view and print the page title, pageview, bounce rate and and average time for the LOWEST PAGE VIEW ONLY
for lind in df.index:
    #check for the page title that has the lowest pageview and the shortest average time on page
    if df['Pageviews'][lind] ==ld and df['Avg. Time on Page'][lind] ==lvd:
        print('Since',df['Page Title'][lind], 'is the least popular page view of ', df['Pageviews'][lind],'and has the lowest view duration of  Average Time \non Page', round(df['Avg. Time on Page'][lind]//60),':',round(df['Avg. Time on Page'][lind]-(df['Avg. Time on Page'][lind]//60)*60), 'minutes.')
        print("Suggestion would be to delete this page from the website as it is not viewed by majority of the users \nand is redundant to continue to have it as a webpage. "
              "However, if there are some important information \nin this webpage it would be better to combine with other webpages that have more page views and \npage duration.")

#total view duration
sumvd = df['Avg. Time on Page'].sum()
print(round(sumvd//60),'.',round(sumvd-(sumvd//60)*60), 'minutes is the sum of  average time across all the pages for the E-Commerce website.')

#see all datas across all pages
#print("\nFor all information regarding each page title, please see the below content.")
#print the data content that is extracted from csv in the form of paragraph(include page title, page views, bounce rate and average time)For average time I convert it from seconds to minutes and seconds
for ind in df.index:
    #check page title
    if df['Page Title'][ind] == 'CART |E-Commerce':
      d = df['Pageviews'][ind]
    #check page title
    elif df['Page Title'][ind] == 'Payment |E-Commerce':
        pagesum = d + df['Pageviews'][ind]
        percentdiff = d/pagesum*100
        print(df['Page Title'][ind], 'page views is', round(percentdiff,2), '% lower than Cart |E-commerce page view and a bounce \nrate of', df['Bounce Rate'][ind],'.User stay at the page with an average time of', round(df['Avg. Time on Page'][ind]//60),':',round(df['Avg. Time on Page'][ind]-(df['Avg. Time on Page'][ind]//60)*60), 'minutes.')
        print("These are some methods to encourage buyers to stay on the payment page which are to introduce \nlast minute discount code for the users to use, introduce a loyalty reward program to encourage \ncustomer to continue to shop and lastly to have a pop up message when the user is about to leave \nthe webpage.")
print("\nSection 2 Conversion Rate")
#Find the highest percentage of goal conversion rate and print out the result
hc= co['goalConversionRateAll'].max()
#find the dates that matches the highest percentage of goal conversion rate
for cind in co.index:
    #check for the dates that have the highest conversion rates percentage and list them out
    if co['goalConversionRateAll'][cind] == hc:
        print (co['date'][cind], 'has the highest conversion rate of', co['goalConversionRateAll'][cind],'%.' )

print("\nSection 3 Users Location")
#find the highest and lowest number of users
hu= lo['Users'].max()
lu= lo['Users'].min()

#print the highest and lowest users and country for the location data
for usercount in lo.index:
    #check which country has the highest user count
    if lo['Users'][usercount] ==hu:
        print('The website mainly target local customers from ',lo['Country'][usercount], 'which has ', lo['Users'][usercount], 'users that has initiated at \nleast one session on the website.')
        print("The website can use location-based marketing whereby to provide sending personalized newsletter \nto the user. Another way is to customised to the experience for the local buyer by including paylah or \npaynow as a payment method which is very common for local buyers."
              "Moreover market research \ncan be conducted such as surveys,etc to better understand the local consumer market. ")
    #check which country has the lowest user count
    elif lo['Users'][usercount] ==lu:
        print('The lowest number of users is from ',lo['Country'][usercount], 'which has ', lo['Users'][usercount], 'users that has initiated at least one  \nsession on the website.')

#find the highest average session duration and convert from seconds to minutes and seconds
hl= lo['Avg. Session Duration'].max()
def convert(seconds):
    return time.strftime("%M:%S", time.gmtime(hl))

#Use for loop to print the list of country data with field such as name of country, number of users, average session duration that is
#converted to minutes and seconds and round off to the nearest seconds
#for lind in lo.index:
    #print('For',lo['Country'][lind], ', the number of user who have initiated at least one session on the website is',lo['Users'][lind],'.\nThe average session'
        #' duration whereby user have spent on the page is', round(lo['Avg. Session Duration'][lind]//60),'.',round(lo['Avg. Session Duration'][lind]-(lo['Avg. Session Duration'][lind]//60)*60),'minutes.', '\nThe total number of session for', lo['Country'][lind],
          #'is', lo['Sessions'][lind],'.')
#print the highest average session duration based on country data
print(convert(hl), "minutes is the highest average session duration for this website.")

#traffic source section
print("\nSection 4 Traffic Sources")
#find channel with highest and lowest users + new users
htu = ts['users'].max()
htnu = ts['newUsers'].max()
ltu = ts['users'].min()
ltnu = ts['newUsers'].min()
#print the highest and lowest users + new users from the traffic source channels.
for t in ts.index:
    #check which traffic source has the highest users and new users
    if ts['users'][t] == htu and ts['newUsers'][t] == htnu:
        print(ts['channelGrouping'][t], "has both the highest users of", ts['users'][t], "and new users of", ts['newUsers'][t], ". Therefore it is a better performing \nchannel in driving more users.")
    #check which traffic source has the lowest user and new users
    elif ts['users'][t] == ltu or ts['newUsers'][t] == ltnu:
        print(ts['channelGrouping'][t], "has either the lowest users of", ts['users'][t], "or new users of", ts['newUsers'][t], ".Therefore it is a not quite effective in \ndriving more users.")
#for tts in ts.index:
    #print("For traffic source from", ts['channelGrouping'][tts], "it has drive", ts['users'][tts],"users and", ts['newUsers'][tts],"new users to the website. It has the \ntotal session of",ts['sessions'][tts],"and at the same time pageviews per session of", round(ts['pageviewsPerSession'][tts], 2),".")

print("\nSection Social Network Traffic ")

#print the list of social network data in text format with social network, percentage of session and average session duration
highs = sn['Sessions'].max()
lows = sn['Sessions'].min()
sumsocial = sn['Sessions'].sum()

#print the highest and lowest sessions from social network
for sessionscount in sn.index:
    #check which social network has the highest sessions and print the information for the highest session
    if sn['Sessions'][sessionscount] == highs:
        print('The highest sessions for social network is from', sn['Social Network'][sessionscount], 'which has', round((sn['Sessions'][sessionscount]/sumsocial)*100, 2 ),'% of sessions where \nby users are actively engaged with the website. The average session duration for', sn['Social Network'][sessionscount], 'is\n', round(sn['Avg. Session Duration'][sessionscount]//60),':', round(sn['Avg. Session Duration'][sessionscount]-(sn['Avg. Session Duration'][sessionscount]//60)*60),'minutes.' )
        print('Since', sn['Social Network'][sessionscount], 'is the most effective social network to drive traffic to the website, we should upload \nmore posts and constantly do research to keep up the with the latest social media trend so that can \nattract more potential users to the website. It will be even better to use advertisement on the social \nmedia platform or some of the business tools they offer such as Instagram shop.')
    #check which social network has the lowest sessioms and print the information and advice for it
    elif sn['Sessions'][sessionscount] == lows:
        print('The lowest sessions for social network is from', sn['Social Network'][sessionscount], 'which has', round((sn['Sessions'][sessionscount]/sumsocial)*100, 2 ),'% of sessions where \nby users are actively engaged with the website. The average session duration for', sn['Social Network'][sessionscount], 'is\n', round(sn['Avg. Session Duration'][sessionscount]//60),':', round(sn['Avg. Session Duration'][sessionscount]-(sn['Avg. Session Duration'][sessionscount]//60)*60),'minutes.' )
        print('Since', sn['Social Network'][sessionscount], 'is the least effective social network to drive traffic to the website, some \nrecommendation to improve would be to schedule posts more frequently, join some Facebook \ngroups to promote the website and conduct livestream occassionally.')

#user retention
print("\nSection 5  User Retention")
hur = ur['Total Users'].max()
for userretent in ur.index:
    #check if the total users from the specific period matches has the highest total users
    if ur['Total Users'][userretent] == hur:
        print (ur['Metrics'][userretent],"has the highest user retention of", ur['Total Users'][userretent],".")
for topretent in ur.index:
    # check which period has a total user retention of more than 0
    if ur['Total Users'][topretent] >0:
        print ("From",ur['Metrics'][topretent],"has a user retention of", ur['Total Users'][topretent]," which is more than 0 compared to the \nother periods.")

#new and returning visitors
print("\nSection 6 New and Returning Visitors")
hnru = nr['users'].max()
hnrnewuser = nr["newUsers"].max()
lnru = nr['users'].min()
lnrnewuser = nr["newUsers"].min()
for nrusers in nr.index:
    #check if new or returning visitors has the highest users and new users for the website
    if nr['users'][nrusers] == hnru and nr['newUsers'][nrusers] == hnrnewuser:
        print(nr['userType'][nrusers],"has both the highest number of users of", nr['users'][nrusers],"and new users of", nr['newUsers'][nrusers],"visiting the website.\n"
        "Even though the session is ", nr['sessions'][nrusers],"which is lower compared to Returning Visitors, it has a higher \npages/sessions view of", round(nr['pageviewsPerSession'][nrusers], 2),". For the average session duration, it is",
        round(nr['avgSessionDuration'][nrusers]//60),':',round(nr['avgSessionDuration'][nrusers]-(nr['avgSessionDuration'][nrusers]//60)*60), 'minutes.' )
    #check if new or returning visitors has the lowest users and new users for the website
    elif nr['users'][nrusers] == lnru and nr['newUsers'][nrusers] == lnrnewuser:
        print(nr['userType'][nrusers],"has of users of", nr['users'][nrusers],"and new users of", nr['newUsers'][nrusers],"visiting the website.\n"
        "Even though the session is ", nr['sessions'][nrusers],"which is higher compared to New Visitors, it has a lower \npages/sessions view of", round(nr['pageviewsPerSession'][nrusers], 2),". For the average session duration, it is",
        round(nr['avgSessionDuration'][nrusers]//60),':',round(nr['avgSessionDuration'][nrusers]-(nr['avgSessionDuration'][nrusers]//60)*60), 'minutes.' )
#browser
print("\nSection 7 Technology details of browsers used")
pbr= br['sessions'].max()
lpbru = br['users'].min()
lpbrnu = br['newUsers'].min()
lpbs = br['sessions'].min()
for brcount in br.index:
    #check fot most popular session for the specific browser
    if br['sessions'][brcount] == pbr:
        print(br['browser'][brcount], "is the most popular browser as such the website or mobile application should be well-tested \nfor",br['browser'][brcount],".",br['browser'][brcount],"manage to drive", br['users'][brcount],"users and",br['newUsers'][brcount],"new users to the website which is the highest \ncompared to other browsers."
              "The session for",br['browser'][brcount],"is",br['sessions'][brcount],"which is also the highest sessions.")

for lp in br.index:
    #check for least popular users and new users
    if br['users'][lp] == lpbru and br['newUsers'][lp] == lpbrnu:
        #check for least popular sessions
        if br['sessions'][lp] == lpbs:
            print(br['browser'][lp], "is the least popular browser as such the website or mobile application is less likely \nfor users to use it on",br['browser'][lp],".",br['browser'][lp],"manage to drive", br['users'][lp],"users and",br['newUsers'][lp],"new users to \nthe website which is the lowest compared to other browsers."
              "\nThe session for",br['browser'][lp],"is",br['sessions'][lp],"which is also the lowest sessions.")
        #if the browser is not the least popular sessions
        else:
            print(br['browser'][lp], "is the least popular browser as such the website or mobile application is less likely for users to \nuse it on",br['browser'][lp],".",br['browser'][lp],"manage to drive", br['users'][lp],"users and",br['newUsers'][lp],"new users to the website which is the \nlowest compared to other browsers."
              "The session for",br['browser'][lp],"is",br['sessions'][lp],".")
#close the text file after all these content has been successfully added to the text file
sys.stdout.close()
