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

print("Page View and View Duration Section")

#Finding the highest page view data and print the result out
d=df['Pageviews'].max()
#Finding the lowest page view data and print the result out
ld=df['Pageviews'].min()

#Use for loop to find the page title that match the highest page view and print the page title, pageview, bounce rate and and average time for the HIGHEST PAGE VIEW ONLY
for ind in df.index:
    if df['Pageviews'][ind] == d:
         print(df['Page Title'][ind], 'has the most popular page view of ', df['Pageviews'][ind],'.')

#Use for loop to find the page title that match the lowest page view and print the page title, pageview, bounce rate and and average time for the LOWEST PAGE VIEW ONLY
for lind in df.index:
    if df['Pageviews'][lind] ==ld:
        print(df['Page Title'][lind], 'is the least popular page view of ', df['Pageviews'][lind],'.')

#find the highest view duration
hvd = df['Avg. Time on Page'].max()
for ind in df.index:
    if df['Avg. Time on Page'][ind] ==hvd:
        print(df['Page Title'][ind], 'has the highest view duration of  Average Time on Page', round(df['Avg. Time on Page'][ind]//60),'.',round(df['Avg. Time on Page'][ind]-(df['Avg. Time on Page'][ind]//60)*60), 'minutes.')

#find the lowest view duration
lvd = df['Avg. Time on Page'].min()
for ind in df.index:
    if df['Avg. Time on Page'][ind] ==lvd:
        print(df['Page Title'][ind], 'has the lowest view duration of  Average Time on Page', round(df['Avg. Time on Page'][ind]//60),'.',round(df['Avg. Time on Page'][ind]-(df['Avg. Time on Page'][ind]//60)*60), 'minutes.')

#total view duration
sumvd = df['Avg. Time on Page'].sum()
print(round(sumvd//60),'.',round(sumvd-(sumvd//60)*60), 'minutes is the sum of  average time across all the pages for the E-Commerce website.')

#see all datas across all pages
print("For all information regarding each page title, please see the below content.")
#print the data content that is extracted from csv in the form of paragraph(include page title, page views, bounce rate and average time)For average time I convert it from seconds to minutes and seconds
for ind in df.index:
    print(df['Page Title'][ind], 'has a page view of ', df['Pageviews'][ind], 'and a bounce rate of', df['Bounce Rate'][ind],'.\nUser stay at the page with an average time of', round(df['Avg. Time on Page'][ind]//60),'.',round(df['Avg. Time on Page'][ind]-(df['Avg. Time on Page'][ind]//60)*60), 'minutes.')

print("\nConversion Rate Section")
#Find the highest percentage of goal conversion rate and print out the result
hc= co['goalConversionRateAll'].max()
print(hc, "% is the highest percentage of conversion rate.")

#find the dates that matches the highest percentage of goal conversion rate
for cind in co.index:
    if co['goalConversionRateAll'][cind] == hc:
        print (co['date'][cind], 'has the highest conversion rate of', co['goalConversionRateAll'][cind],'%.' )
print("\nUsers Location Section")
#find the highest and lowest number of users
hu= lo['Users'].max()
lu= lo['Users'].min()

#print the highest and lowest users and country for the location data
for usercount in lo.index:
    if lo['Users'][usercount] ==hu:
        print('The highest number of users is from ',lo['Country'][usercount], 'which has ', lo['Users'][usercount], 'users that has initiated at least one \nsession on the website.')
    elif lo['Users'][usercount] ==lu:
        print('The lowest number of users is from ',lo['Country'][usercount], 'which has ', lo['Users'][usercount], 'users that has initiated at least one  \nsession on the website.')

#find the highest average session duration and convert from seconds to minutes and seconds
hl= lo['Avg. Session Duration'].max()
def convert(seconds):
    return time.strftime("%M:%S", time.gmtime(hl))

#Use for loop to print the list of country data with field such as name of country, number of users, average session duration that is
#converted to minutes and seconds and round off to the nearest seconds
for lind in lo.index:
    print('For',lo['Country'][lind], ', the number of user who have initiated at least one session on the website is',lo['Users'][lind],'.\nThe average session'
        ' duration whereby user have spent on the page is', round(lo['Avg. Session Duration'][lind]//60),'.',round(lo['Avg. Session Duration'][lind]-(lo['Avg. Session Duration'][lind]//60)*60),'minutes.', '\nThe total number of session for', lo['Country'][lind],
          'is', lo['Sessions'][lind],'.')
#print the highest average session duration based on country data
print(convert(hl), "minutes is the highest average session duration for this website.")

#traffic source section
print("\nTraffic Sources Section")
#find channel with highest and lowest users + new users
htu = ts['users'].max()
htnu = ts['newUsers'].max()
ltu = ts['users'].min()
ltnu = ts['newUsers'].min()
#print the highest and lowest users + new users from the traffic source channels.
for t in ts.index:
    if ts['users'][t] == htu and ts['newUsers'][t] == htnu:
        print(ts['channelGrouping'][t], "has both the highest users of", ts['users'][t], "and new users of", ts['newUsers'][t], ". Therefore it is a better performing \nchannel in driving more users.")
    elif ts['users'][t] == ltu or ts['newUsers'][t] == ltnu:
        print(ts['channelGrouping'][t], "has either the lowest users of", ts['users'][t], "or new users of", ts['newUsers'][t], ".Therefore it is a not quite effective in \ndriving more users.")
for tts in ts.index:
    print("For traffic source from", ts['channelGrouping'][tts], "it has drive", ts['users'][tts],"users and", ts['newUsers'][tts],"new users to the website. It has the \ntotal session of",ts['sessions'][tts],"and at the same time pageviews per session of", round(ts['pageviewsPerSession'][tts], 2),".")

print("\nSocial Network Traffic Section")
#print the list of social network data in table format
print(sn)

#print the list of social network data in text format with social network, percentage of session and average session duration
highs = sn['Sessions'].max()
lows = sn['Sessions'].min()
sumsocial = sn['Sessions'].sum()

#print the highest and lowest sessions from social network
for sessionscount in sn.index:
    if sn['Sessions'][sessionscount] == highs:
        print('The highest sessions for social network is from', sn['Social Network'][sessionscount], 'which has', round((sn['Sessions'][sessionscount]/sumsocial)*100, 2 ),'% of sessions where \nby users are actively engaged with the website.' )
    elif sn['Sessions'][sessionscount] == lows:
        print('The lowest sessions for social network is from', sn['Social Network'][sessionscount], 'which has', round((sn['Sessions'][sessionscount]/sumsocial)*100, 2 ),'% of sessions where \nby users are actively engaged with the website.' )

for sind in sn.index:
    print(sn['Social Network'][sind], 'has a percentage of ',round((sn['Sessions'][sind]/sumsocial)*100, 2 ),'% of sessions where by users are actively engaged with \nthe website. The average session duration for', sn['Social Network'][sind], 'is', round(sn['Avg. Session Duration'][sind]//60),'.', round(sn['Avg. Session Duration'][sind]-(sn['Avg. Session Duration'][sind]//60)*60),'minutes.')

#user retention
print("\nUser Retention Section")
hur = ur['Total Users'].max()
for userretent in ur.index:
    if ur['Total Users'][userretent] == hur:
        print (ur['Metrics'][userretent],"has the highest user retention of", ur['Total Users'][userretent],".")
for topretent in ur.index:
    if ur['Total Users'][topretent] >0:
        print ("From",ur['Metrics'][topretent],"has a user retention of", ur['Total Users'][topretent]," which is more than 0 compared to the \nother periods.")

#new and returning visitors
print("\nNew and Returning Visitors Section")
hnru = nr['users'].max()
hnrnewuser = nr["newUsers"].max()
lnru = nr['users'].min()
lnrnewuser = nr["newUsers"].min()
for nrusers in nr.index:
    if nr['users'][nrusers] == hnru and nr['newUsers'][nrusers] == hnrnewuser:
        print(nr['userType'][nrusers],"has both the highest number of users of", nr['users'][nrusers],"and new users of", nr['newUsers'][nrusers],"visiting the website.\n"
        "Even though the session is ", nr['sessions'][nrusers],"which is lower compared to Returning Visitors, it has a higher \npages/sessions view of", round(nr['pageviewsPerSession'][nrusers], 2),". For the average session duration, it is",
        round(nr['avgSessionDuration'][nrusers]//60),'.',round(nr['avgSessionDuration'][nrusers]-(nr['avgSessionDuration'][nrusers]//60)*60), 'minutes.' )
    elif nr['users'][nrusers] == lnru and nr['newUsers'][nrusers] == lnrnewuser:
        print(nr['userType'][nrusers],"has of users of", nr['users'][nrusers],"and new users of", nr['newUsers'][nrusers],"visiting the website.\n"
        "Even though the session is ", nr['sessions'][nrusers],"which is higher compared to New Visitors, it has a lower \npages/sessions view of", round(nr['pageviewsPerSession'][nrusers], 2),". For the average session duration, it is",
        round(nr['avgSessionDuration'][nrusers]//60),'.',round(nr['avgSessionDuration'][nrusers]-(nr['avgSessionDuration'][nrusers]//60)*60), 'minutes.' )
#browser
print("\nTechnology details of browsers used")
pbr= br['sessions'].max()
lpbru = br['users'].min()
lpbrnu = br['newUsers'].min()
lpbs = br['sessions'].min()
for brcount in br.index:
    if br['sessions'][brcount] == pbr:
        print(br['browser'][brcount], "is the most popular browser as such the website or mobile application should be well-tested \nfor",br['browser'][brcount],".",br['browser'][brcount],"manage to drive", br['users'][brcount],"users and",br['newUsers'][brcount],"new users to the website which is the highest \ncompared to other browsers."
              "The session for",br['browser'][brcount],"is",br['sessions'][brcount],"which is also the highest sessions.")
for lp in br.index:
    if br['users'][lp] == lpbru and br['newUsers'][lp] == lpbrnu:
        if br['sessions'][lp] == lpbs:
            print(br['browser'][lp], "is the least popular browser as such the website or mobile application is less likely \nfor users to use it on",br['browser'][lp],".",br['browser'][lp],"manage to drive", br['users'][lp],"users and",br['newUsers'][lp],"new users to \nthe website which is the lowest compared to other browsers."
              "\nThe session for",br['browser'][lp],"is",br['sessions'][lp],"which is also the lowest sessions.")
        else:
            print(br['browser'][lp], "is the least popular browser as such the website or mobile application is less likely for users to \nuse it on",br['browser'][lp],".",br['browser'][lp],"manage to drive", br['users'][lp],"users and",br['newUsers'][lp],"new users to the website which is the \nlowest compared to other browsers."
              "The session for",br['browser'][lp],"is",br['sessions'][lp],".")
#close the text file after all these content has been successfully added to the text file
sys.stdout.close()
