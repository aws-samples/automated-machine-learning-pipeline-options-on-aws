import sys
from datetime import date
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import when
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.sql.functions import col, expr, when, round
from pyspark.sql.types import LongType
from awsglue.dynamicframe import DynamicFrame

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'PROCESSED_DIR', 'INPUT_DIR'])

processed_dir = args['PROCESSED_DIR']
input_dir = args['INPUT_DIR']

job.init(args['JOB_NAME'], args)

#database = 'iris-database' #replace with your user id
today = date.today()
logger = glueContext.get_logger()
logger.info("info message")


df= glueContext.create_dynamic_frame_from_options(format_options={"quoteChar": '"', "withHeader": True, "separator": ","},connection_type = "s3", connection_options = {"paths": [input_dir]}, format = "csv")

df1 = df.toDF()
#print(df1)

#drop the columns that we will not use
cols = ("Day_Charge", "Eve_Charge", "Night_Charge", "Intl_Charg", "Area_Code","State", "sentiment")

data=df1.drop(*cols) 
data.printSchema()


#change the column type
from pyspark.sql.types import StringType, DateType, FloatType, LongType, BooleanType

data2= data \
  .withColumn("Account_Length" ,
              data["Account_Length"]
              .cast(LongType()))  \
  .withColumn("customerID",
              data["customerID"]
              .cast(LongType()))    \
  .withColumn("Int_l_Plan"  ,
              data["Int_l_Plan"]
              .cast(StringType())) \
  .withColumn("VMail_Plan"  ,
              data["VMail_Plan"]
              .cast(StringType())) \
  .withColumn("VMail_Message"  ,
              data["VMail_Message"]
              .cast(StringType())) \
  .withColumn("Day_Mins"  ,
              data["Day_Mins"]
              .cast(FloatType())) \
  .withColumn("Day_Calls"  ,
              data["Day_Calls"]
              .cast(LongType())) \
  .withColumn("Eve_Mins"  ,
              data["Eve_Mins"]
              .cast(FloatType())) \
  .withColumn("Eve_Calls"  ,
              data["Eve_Calls"]
              .cast(LongType())) \
  .withColumn("Night_Mins"  ,
              data["Night_Mins"]
              .cast(FloatType())) \
  .withColumn("Night_Calls"  ,
              data["Night_Calls"]
              .cast(LongType())) \
  .withColumn("Intl_Mins"  ,
              data["Intl_Mins"]
              .cast(FloatType())) \
  .withColumn("Intl_Calls"  ,
              data["Intl_Calls"]
              .cast(LongType())) \
.withColumn("CustServ_Calls"  ,
              data["CustServ_Calls"]
              .cast(LongType())) \
.withColumn("Churn"  ,
              data["Churn"]
              .cast(BooleanType())) \
.withColumn("pastSenti_nut"  ,
              data["pastSenti_nut"]
              .cast(LongType())) \
.withColumn("pastSenti_pos"  ,
              data["pastSenti_pos"]
              .cast(LongType())) \
.withColumn("pastSenti_neg"  ,
              data["pastSenti_neg"]
              .cast(LongType())) \
.withColumn("mth_remain"  ,
              data["mth_remain"]
              .cast(LongType())) \
  
      
data2.printSchema()

from pyspark.sql import functions as F

data3 = data2.withColumn('Churn', F.when(data2.Churn == 'false', 0).otherwise(1))

data4 = data3.withColumn('Int_l_Plan', F.when(data3.Int_l_Plan == 'no', 0).otherwise(1))

data5 = data4.withColumn('VMail_Plan', F.when(data4.VMail_Plan == 'no', 0).otherwise(1))

data5.printSchema()
data5.select('churn').show()

data_final=data5.select("churn", 
 'Account_Length',
 'customerID',
 'Int_l_Plan',
 'VMail_Plan',
 'VMail_Message',
 'Day_Mins',
 'Day_Calls',
 'Eve_Mins',
 'Eve_Calls',
 'Night_Mins',
 'Night_Calls',
 'Intl_Mins',
 'Intl_Calls',
 'Intl_Charge',
 'CustServ_Calls',
 'pastSenti_nut',
 'pastSenti_pos',
 'pastSenti_neg',
 'mth_remain')


df_pandas = data_final.toPandas()
val = df_pandas.sample(frac=0.2, axis=0)
train_df = df_pandas.drop(index=val.index)
test_df = val.sample(frac=0.05, axis=0)
val_df = val.drop(index=test_df.index)


train_dir=processed_dir+"train/train.csv"
val_dir=processed_dir+"validation/validation.csv"
test_dir=processed_dir+"test/test.csv"

train_df.to_csv(train_dir, index=False, line_terminator="")
val_df.to_csv(val_dir, index=False, header=False, line_terminator="")
test_df.to_csv(test_dir, index=False, header=False, line_terminator="")

job.commit()
