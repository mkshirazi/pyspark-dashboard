
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/home/mmkshira/spark-3.2.0-bin-hadoop2.7"

import findspark
findspark.init()


from pyspark.sql import SparkSession
import random

spark = SparkSession.builder.appName("YourTest").master("local[2]").config('spark.ui.port', random.randrange(4000,5000)).getOrCreate()
from pyspark.sql.functions import explode
from pyspark.sql import functions as F

sc = spark.sparkContext
sc.setLogLevel("ERROR")

import pandas as pd
import plotly.express as px
from datetime import datetime


### Distinct topics
def distinct_topics():
    return spark.sql("select distinct topic from topics")
# Top papers by topic
def top_papers(n=5):
    return main_topic_df.select("id","papertitle","authors","year","citationcount")\
        .orderBy("citationcount", ascending=False).limit(n)\
        .select("id","papertitle", explode("authors").alias("author"),"year","citationcount")\
        .select("id","papertitle","year","citationcount", F.col("author").getItem("name").alias("name"))\
        .groupBy(["id","papertitle","year","citationcount"]).agg(F.collect_list("name").alias("authors"))#.show()

# For timeline chart. number of citations. Each dot represents a paper
def timeline():
    df = main_topic_df.select("id","year","citationcount")\
        .orderBy("year").toPandas()
    fig = px.scatter(df,x="year",y="citationcount", opacity=0.5)
    return fig


#Top authors for a topic
def top_authors(n=5):
    global exploded_authors_topic
    exploded_authors_topic = main_topic_df.select("id",explode("authors").alias("author"),"citationcount")\
        .select("id", \
            F.col("author").getItem("country").alias("country"), \
            F.col("author").getItem("name").alias("name"), \
            F.col("author").getItem("id").alias("author_id"), \
            F.col("author").getItem("affiliation").alias("affiliation"), \
            "citationcount").cache()
    return exploded_authors_topic.select("name","author_id","citationcount")\
        .groupBy(["name","author_id"]).agg(F.sum("citationcount").alias("citationcount"))\
        .filter("name is not null")\
        .orderBy("citationcount", ascending=False).limit(n)
#Top institutions for a topic
def top_institutions(n=5):
    return exploded_authors_topic.select("id","affiliation","citationcount").distinct()\
        .groupBy(["affiliation"]).agg(F.sum("citationcount").alias("citationcount"))\
        .filter("affiliation is not null")\
        .orderBy("citationcount", ascending=False).limit(n)
def by_type():
    df = main_topic_df.select("id","type").distinct()\
        .filter("type is not null")\
        .groupBy("type").agg(F.count("id").alias("number of papers")).toPandas()
    return px.pie(df,names="type",values="number of papers")#.show()
     
# countries of institutions
def countries():
    df = exploded_authors_topic.select("id","country","citationcount").distinct()\
            .groupBy("country").agg(F.sum("citationcount").alias("citations")).toPandas()
    map_data = px.data.gapminder().query("year==2007")[["iso_alpha","country","continent"]]
    # df
    map_data = df[1:].merge(map_data,how="left").dropna()
    fig = px.choropleth(map_data, locations="iso_alpha",color="citations",
                     hover_name="country",color_continuous_scale=px.colors.sequential.Plasma)
    return fig#fig.show()

def main_topic(sel_topic, n_paper=5, n_author=5, n_institution=5):
    global main_topic_df
    main_topic_df = all_data_sql.filter(F.array_contains("topics",sel_topic)).select("id","type","papertitle","authors",F.year("year").alias("year"),"citationcount").cache()
    # Need to maintain this ordering to set global variables correctly
    top_author = top_authors(n_author)
    top_institutes = top_institutions(n_institution)
    top_paper = top_papers(n_paper)
    timeline_data = timeline()
    type_data = by_type()
    countries_data = countries()


    return top_author, top_institutes, top_paper, timeline_data, type_data, countries_data


files_to_read = ["sample.jsonl", "1000_lines.json", "10000_lines.json", "100000_lines.json", "test.json", "2_million_lines.json"]

time =[]
size =[]
for file in files_to_read:
    start_time = datetime.now()
    training_data = spark.read.json(file)
    training_data.createOrReplaceTempView("all_data")
    all_data_sql = spark.sql("select * from all_data")

    top_author, top_institutes, top_paper, timeline_data, type_data, countries_data = main_topic("computer science")
    #top_author.show()
    #top_institutes.show()
    #top_paper.show()
    #timeline_data.show()
    #type_data.show()
    #countries_data.show()

    time_taken = datetime.now() - start_time
    file_size = all_data_sql.count()
    spark.catalog.clearCache()
    print("file size: " + str(file_size))
    print("file: " + file)

    float_time_taken = float(time_taken.total_seconds())
    print("Time taken: " + str(float_time_taken) + " seconds")

    time.append(float_time_taken)
    size.append(file_size)
    
from matplotlib import pyplot as plt
print(size)
print(time)
plt.plot(size[1:],time[1:])
plt.xlabel("Number of papers \n (1 million papers have a size of 3GB)")
plt.ylabel("Time taken (seconds)")
plt.title("Performance scalability")
plt.savefig("performance.png")
plt.show()

    
""" # correlated topics with selected topics
sel_topic = "artificial intelligence"
testing2 = main_topic_df.filter(F.array_contains("topics",sel_topic)).select("id",explode("topics").alias("topic")).filter("topic != \'" +sel_topic+ "\'")
testing3 = testing2.groupBy("topic").count().sort("count",ascending=False).show()
testing2.show()

# Authors for a given topic
sel_topic = "multimedia"
testing2 = testing1.filter(F.array_contains("topics",sel_topic)).select("id",explode("authors").alias("authors"),"topics","type")
testing3 = testing2.select(testing2.id.alias("paper_id"), testing2.topics.alias("topics"), testing2.type.alias("type"), \
    F.col("authors").getItem("name").alias("name"), \
    F.col("authors").getItem("country").alias("country"), \
    F.col("authors").getItem("id").alias("author_id"), \
    F.col("authors").getItem("affiliation").alias("affiliation"), \
    )
testing4 = testing3.groupBy("country").count().sort("count",ascending=False) # country for a given topic
testing4 = testing3.groupBy("type").count().sort("count",ascending=False) # type for a given topic
testing4 = testing3.groupBy("affiliation").count().sort("count",ascending=False) # affiliation for a given topic
testing4 = testing3.groupBy("name").count().sort("count",ascending=False) # authors for a given topic
testing4.show(5,False)


# Calculate h-index

# Conferences for a given topic
sel_topic = "computer science"
testing2 = testing1.filter(F.array_contains("topics",sel_topic)).select("id", \
    "confname","type", 'conferenceseriesid', 'confname', 'confplace', 'confseries', 'confseriesname') \
    .filter("confname is not null")


testing3 = testing2.groupBy("confseriesname").count().sort("count",ascending=False) # confseries for a given topic
testing3 = testing2.groupBy("confname").count().sort("count",ascending=False)
testing3.show()

# Citations
sel_topic = "multimedia"
testing2 = testing1.filter(F.array_contains("topics",sel_topic)).select("id",explode("industrial_sectors").alias("industrial_sector")).filter("industrial_sectors is not null")
testing2.show(10,False)

 """