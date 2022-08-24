'''
 name_file_field = pd.read_csv(path_field + filename2)

    fig1, ax1 = plt.subplots()
    if option_field == 'Show me total ratio for the chosen month' and full_ratio_field is False:
        st.markdown('Please check the monthly statistics check box to choose a month.')

    elif option_field == 'Show me total ratio for the chosen month' and full_ratio_field:
        name_file = pd.read_csv(path + filename)

        gender_array = name_file['gender']
        new_gender_arr = [gender for gender in gender_array if gender != 'andy' and gender != 'unknown']

        f_count = 0
        for elt in new_gender_arr:
            if elt == 'female' or elt == 'mostly_female':
                f_count += 1

        m_count = len(new_gender_arr) - f_count
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = "female", "male"
        sizes = [f_count / (f_count + m_count), m_count / (f_count + m_count)]

        # fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)

    elif option_field != 'Show me total ratio for the chosen month' and full_ratio_field:

        new_data = name_file_field[name_file_field['field'] == option_field]

        if len(new_data) == 0:
            st.caption(
                'This field does not exist for the month of ' + option + '. If you want to see overall ratio for the
                 month of January, please uncheck the box to see the total ratio for the field or choose another field.')
        else:
            new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

            # choosing the selected field from the dataframe
            f_field_count = int(new_data[new_data['variable'] == 'female']['value'])
            m_field_count = int(new_data[new_data['variable'] == 'male']['value'])

            field_size = new_data['value'].sum()
            sizes_field = [f_field_count / field_size, m_field_count / field_size]
            labels = "female", "male"

            # fig1, ax1 = plt.subplots()
            ax1.pie(sizes_field, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
    else:
        st.markdown('Please choose a field or click on check box to see an analysis.')
        #filename = 'combined_ratio_fields.csv'
        name_file = pd.read_csv(path_field + filename)
        f_count = name_file[name_file['field'] == option_field][name_file['variable'] == 'female']['value'].astype(int)
        m_count = name_file[name_file['field'] == option_field][name_file['variable'] == 'male']['value'].astype(int)
        #new_data = name_file_field[name_file_field['field'] == option_field]
        #new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

        # choosing the selected field from the dataframe
        #f_field_count = new_data[new_data['variable'] == 'female']['value'].astype(int)
        #m_field_count = new_data[new_data['variable'] == 'male']['value'].astype(int)

        #field_size = new_data['value'].sum()ü)
        field_size = f_count +m_count
        sizes_field = [f_count / field_size, m_count / field_size]
        labels = "female", "male"

        # fig1, ax1 = plt.subplots()
        ax1.pie(sizes_field, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)











# Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = "female", "male"
    sizes = [f_count / len(new_gender_arr), m_count / len(new_gender_arr)]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            ###some data processing to collect the information of female and male quoting in articles acc.to fields

        gender_list = {'female': [], 'male': [], 'unknown': [], 'andy': []}
        for i in range(len(new_data)):
            text = new_data['content'].iloc[i]
            doc = nlp(text)
            count_m = 0
            count_f = 0
            count_a = 0
            count_u = 0
            for elt in doc.ents:
                if elt.label_ == 'PERSON':
                    if d.get_gender(str(elt[0]).rsplit(None, 1)[0]) == 'female':
                        count_f += 1
                    if d.get_gender(str(elt[0]).rsplit(None, 1)[0]) == 'male':
                        count_m += 1
                    if d.get_gender(str(elt[0]).rsplit(None, 1)[0]) == 'unknown':
                        count_u += 1
                    if d.get_gender(str(elt[0]).rsplit(None, 1)[0]) == 'andy':
                        count_a += 1
            gender_list['female'].append(count_f)
            gender_list['male'].append(count_m)
            gender_list['unknown'].append(count_u)
            gender_list['andy'].append(count_a)

        gender_list_df = pd.DataFrame(gender_list)

        final = (pd.concat([new_data, gender_list_df], axis=1, join='inner'))
        reshaped_df = final.melt(id_vars=['Unnamed: 0', 'web_url', 'content'],
                                 value_vars=['female', 'male', 'unknown', 'andy'])
        df_new = reshaped_df.groupby(['variable'], as_index=False)['value'].sum()
        print(df_new)
        df_gender_final = pd.DataFrame(df_new)
         if option == 'January':
        filename2 = '2022-1._fields.csv'
    elif option == 'February':
        filename2 = '2022-2._fields.csv'
    elif option == 'March':
        filename2 = '2022-3._fields.csv'
    elif option == 'April':
        filename2 = '2022-4._fields.csv'
    elif option == 'May':
        filename2 = '2022-5._fields.csv'
    elif option == 'June':
        filename2 = '2021-6._fields.csv'
    elif option == 'July':
        filename2 = '2021-7._fields.csv'
    elif option == 'August':
        filename2 = '2021-8._fields.csv'
    elif option == 'September':
        filename2 = '2021-9._fields.csv'
    elif option == 'October':
        filename2 = '2021-10_fields.csv'
    elif option == 'November':
        filename2 = '2021-11_fields.csv'
    elif option == 'December':
        filename2 = '2021-12_fields.csv'
    else:
        filename2 = 'combined_ratio_fields.csv



        # else:
               name_file = pd.read_csv(path_field + filename)

               gender_array = name_file['gender']
               new_gender_arr = [gender for gender in gender_array if gender != 'andy' and gender != 'unknown']

               f_count = 0
               for elt in new_gender_arr:
                   if elt == 'female' or elt == 'mostly_female':
                       f_count += 1

               m_count = len(new_gender_arr) - f_count
'''

import os
import pandas as pd
import spacy
import streamlit as st
import matplotlib.pyplot as plt
import gender_guesser.detector as gender

nlp = spacy.load("en_core_web_sm")
d = gender.Detector()
path = r'./headlines/data/names/'
files = os.listdir(path)
# path_field = r'./headlines/data/fields/'

path_field = r'C:/Users/ilayd/IDP/Scripts/headlines/data/fields/'


def app():
    st.header(" Analysing Gender Bias in New York Times")
    st.markdown('The data that has been collected from  New York Times includes the timeframe of June 2021-May 2022.')
    #st.markdown('Currently in this page, we see the ratios of quoting female and male according to a field in a year.')
    filename = ' '
    filename2 = ' '
    option_field = st.selectbox("Choose a field to see ratio of quoting female in that field. ",
                                #"If you choose none of the fields, total ratio of quoting female in a year"
                               # " from all articles will be visualized.(coming soon)",
                                ('Show me the total ratio for the chosen month', 'arts', 'at-home', 'world', 'sports',
                                 'well', 'us',
                                 'technology', 'science', 'business', 'books', 'briefing', 'climate', 'education',
                                 'garden', 'health', 'insider', 'magazine', 'movies',
                                 'obituaries', 'parenting', 'realestate', 'style', 'theater', 'todayspaper',
                                 'travel',
                                 'dining', 'opinion', 'learning', 'podcasts', 'fashion', 'your-money'))
    full_ratio_field = st.checkbox('Let me choose the month to see monthly statistics.')

    if full_ratio_field:
        option = st.selectbox("Choose a month to see ratio of quoting female.",
                              ('January', 'February', 'March', 'April',
                               'May', 'June', 'July', 'August', 'September',
                               'October', 'November', 'December'))
        if option == 'January':
            filename = '2022-1_names.csv'
            filename2 = '2022-1._fields.csv'
        elif option == 'February':
            filename2 = '2022-2._fields.csv'
            filename = '2022-2_names.csv'
        elif option == 'March':
            filename2 = '2022-3._fields.csv'
            filename = '2022-3_names.csv'
        elif option == 'April':
            filename2 = '2022-4._fields.csv'
            filename = '2022-4_names.csv'
        elif option == 'May':
            filename2 = '2022-5._fields.csv'
            filename = '2022-5_names.csv'
        elif option == 'June':
            filename2 = '2021-6._fields.csv'
            filename = '2021-6_names.csv'
        elif option == 'July':
            filename2 = '2021-7._fields.csv'
            filename = '2021-7_names.csv'
        elif option == 'August':
            filename2 = '2021-8._fields.csv'
            filename = '2021-8_names.csv'
        elif option == 'September':
            filename2 = '2021-9._fields.csv'
            filename = '2021-9_names.csv'
        elif option == 'October':
            filename2 = '2021-10_fields.csv'
            filename = '2021-10_names.csv'
        elif option == 'November':
            filename2 = '2021-11_fields.csv'
            filename = '2021-11_names.csv'
        else:
            # option == 'December':
            filename2 = '2021-12_fields.csv'
            filename = '2021-12_names.csv'

        if option_field != 'Show me the total ratio for the chosen month':

            name_file_field = pd.read_csv(path_field + filename2)
            new_data = name_file_field[name_file_field['field'] == option_field]

            if len(new_data) == 0:
                st.caption(
                    'This field does not exist for the month of ' + option +
                    '. If you want to see overall ratio for the month of January please choose Show me'
                    ' whole ratio option or choose another field.')
            else:
                new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

                # choosing the selected field from the dataframe
                f_field_count = int(new_data[new_data['variable'] == 'female']['value'])
                m_field_count = int(new_data[new_data['variable'] == 'male']['value'])

                field_size = new_data['value'].sum()
                sizes_field = [f_field_count / field_size, m_field_count / field_size]
                labels = "female", "male"

                fig1, ax1 = plt.subplots()
                ax1.pie(sizes_field, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')
                st.markdown(
                    'In this chart, we see the ratios in the month of ' + option + ' for the field of ' + option_field+ '.')
                st.pyplot(fig1)
        else:
            # filename2 = 'combined_ratio_fields.csv'
            # filename = 'combined_ratio_fields.csv'
            name_file = pd.read_csv(path + filename)

            gender_array = name_file['gender']
            new_gender_arr = [gender for gender in gender_array if gender != 'andy' and gender != 'unknown']

            f_count = 0
            for elt in new_gender_arr:
                if elt == 'female' or elt == 'mostly_female':
                    f_count += 1

            m_count = len(new_gender_arr) - f_count

            field_size = m_count + f_count
            sizes_field = [f_count / field_size, m_count / field_size]
            labels = "female", "male"

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes_field, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')

            st.pyplot(fig1)

    else:
        if option_field != 'Show me the total ratio for the chosen month':

            filename3 = 'combined_ratio_fields.csv'
            name_file = pd.read_csv(path_field + filename3)
            new_data = name_file[name_file['field'] == option_field]


            new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

            # choosing the selected field from the dataframe
            f_field_count = int(new_data[new_data['variable'] == 'female']['value'])
            m_field_count = int(new_data[new_data['variable'] == 'male']['value'])

            field_size = new_data['value'].sum()
            sizes_field = [f_field_count / field_size, m_field_count / field_size]
            labels = "female", "male"

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes_field, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')
            st.markdown(
                'In this chart, we see the the ratio for the field of ' + option_field + ' in a year.')
            st.pyplot(fig1)

        else:

            st.markdown('Please choose a field or click on check box to see an analysis.')



        # f_count = name_file[name_file['field'] == option_field][name_file['variable'] == 'female']['value'].astype(int)
