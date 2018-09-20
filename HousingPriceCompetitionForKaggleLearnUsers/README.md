# 房价评估

# 数据说明

**MSSubClass**：建筑类型，int64,，[60,20,70,...]，无缺失值

**MSZoning**：一般分区分类,object,[RL,FV,RM,..]，无缺失值

**LotFrontage**:到property的直线距离,float64,[62.0,NaN,...],有缺失值

**LotArea**:平方英尺,int64,无缺失值

**Street**：道路通行类型,object,[Pave],无缺失值

**Alley**：小巷通道类型,object,[Pave,Grvl,NaN,...],有缺失值

**LotShape**:房间的形状，object,[Reg,IR1,...],无缺失值

**LandContour**:地面平整度，object,[Lvl,Bnk,Low,...],无缺失值

**Utilities**：公共事业可用类型，object,[AllPub,...],无缺失值

**LotConfig**：Lot configuration,object,[Inside,FR2,...],无缺失值

**LandSlope**：坡度,object,[Gtl,Mod,...],无缺失值

**Neighborhood**：Ames市的物理位置，object,[CollgCr,Veenker,...],无缺失值

**Condition1**：靠近主要道路或者铁路,object,[Norm,Feedr,..],无缺失值

**Condition2**：靠近主要道路（如果有第二条）,object,无缺失值

**BldgType**：住宅类型,object,[1Fam,2fmCon,...],无缺失值

**HouseStyle**：住宅风格,object,[1Story,SLvl,1.5Fin,...],无缺失值

**OverallCond**：整体材料和成品质量,int64,[5,6,7,...]，无缺失值

*转化为onehot*

**OverallQual**:整体评估,int64,[5,6,7...],无缺失值

*转化为onehot*

**YearBuilt**：建造年份，int64,[1966,2005,...],无缺失值

*前期处理，用最大年份(2012年)-年份*

**YearRemodAdd**:改造日期,int64,无缺失值

*前期处理，最大年份(2010年)-年份*

**RoofStyle**:顶棚特点,object,[Gable,Hip,...],无缺失值

**RoofMatl**:顶棚材料,object,[CompShg],无缺失值

**Exterior1st**:房子外部遮挡物,object,[VinylSd,MetalSd,...],无缺失值

**Exterior2nd**：房子第二种外部遮挡物，object,[HdBoard,VinylSd,...]，无缺失值

**MasVnrType**：砌体贴面类型，object,[BrkFace,None],有缺失值

**MasVnrArea**：砌体单板面积,float64,有缺失值

**ExterQual**:外观材质,object,[TA,Gd,...]，无缺失值

**ExterCond**:外部材料的形状，object,[TA,Gd,...]，无缺失值

**Foundation**:基础类型,object,[CBlock,PConc,...]，无缺失值

**BsmtQual**：地下室高度,object,[Gd,TA,...],有缺失值

**BsmtCond**:地下室的一般情况，object,[TA,Gd,NaN]，有缺失值

**BsmtExposure**：Walkout or garden level basement walls,object,[No,Gd,NaN,...]，有缺失值

**BsmtFinType1**:地下室成品质量，object,有缺失值

**BsmtFinSF1**:地下室完成平方英尺,int64,无缺失值

**BsmtFinType2**：第二个成品区的质量，object,[Unf,ALQ,NaN,...]

**BsmtFinSF2**：第二个成品的完成面积,int64,无缺失值

**BsmtUnfSF**:未完成（装饰？）的地下室面积,int64,无缺失值

**TotalBsmtSF**:地下室总面积,int64,无缺失值

**Heating**:暖气类型,object,[GasA],无缺失值

**HeatingQC**：暖气质量和条件,object,[Ex,TA,...]，无缺失值

**CentralAir**:中央空调,object,[Y,N,...]，无缺失值

**Electrical**：电力系统,object,[SBrkr,FuseA,...],有缺失值

**1stFlrSF**：第一层楼面积,int64,无缺失值

**2ndFlrSF**：第二层楼面积,int64,无缺失值

**LowQualFinSF**：低质量面积,int64,无缺失值

**GrLivArea**：地面以上生活区面积,int64,无缺失值

**BsmtFullBath**：Basement full bathrooms地下室全浴室,int64,[0,1]，无缺失值

**BsmtHalfBath**:Basement half bathrooms地下室半浴室,int64,[0,1],无缺失值

**FullBath**:Full bathrooms above grade,int64,[1,2,3]，无缺失值

**HalfBath**:Half baths above grade,int64,[0,1],无缺失值

**BedroomAbvGr**：地下室以上的卧室数量，int64,[1,2,3,4]，无缺失值

**KitchenAbvGr**:厨房数量,int64,[1,2],无缺失值

**KitchenQual**：厨房质量,object,[TA,Ex,...],无缺失值

**TotRmsAbvGrd**：所有房间(不包括卧室),int64,无缺失值

**Functional**：房间功能性,object,[Min2,Typ,..]，无缺失值

**Fireplaces**:壁炉数量,int64,无缺失值

**FireplaceQu**:壁炉质量,object,[TA,Ex,Gd,NaN,...]，有缺失值

**GarageType**：车库位置,object,[Attchd,Detchd,...],有缺失值

**GarageYrBlt**：车库建造年份,float64,有缺失值

*方法1，前期处理，最大年份(2010年)-年份..缺失值按照最大值计算*

*方法2，前期处理，有装饰和确实的分开*

**GarageFinish**：车库内部装饰,object,[Fin,RFn,...]，有缺失值

**GarageCars**:车库容量,int64,[0,1,2,3],无缺失值

**GarageArea**：车库面积,int64,无缺失值

**GarageQual**：车库质量,objet,[TA,Fa,...],有缺失值

**GarageCond**：车库环境,object,[TA,NaN,..]，有缺失值

**PavedDrive**：铺好的路,object,[Y,P,N,..]，无缺失值

**WoodDeckSF**:木甲板面积,int64，无缺失值

**OpenPorchSF**:开放式门廊面积,int64,无缺失值

**EnclosedPorch**：封闭的门廊面积,int64,无缺失值

**3SsnPorch**：三个季度的门廊面积,int64,无缺失值

**ScreenPorch**：屏风门廊面积,int64,无缺失值

**PoolArea**：游泳池面积,int64,无缺失值

**PoolQC**:游泳池质量,object,有缺失值

**Fence**：围栏质量,object,[GdPrv,MnPrv,...],有缺失值

**MiscFeature**：未在其他类别中覆盖的杂项功能,object,[shed,NaN,...],有缺失值

**MiscVal**：其他功能的价值,int64,无缺失值

**MoSold**：销售的月份,int64,无缺失值

**YrSold**：销售的年份,int64

**SaleType**：销售类型,object,[WD,COD,...]

**SaleCondition**：销售环境,object,[Abnorml,Partial,...],无缺失值









