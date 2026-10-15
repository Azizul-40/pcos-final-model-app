# PCOS Risk Screening — Streamlit Prototype

এটা আমাদের চূড়ান্ত pipeline-এর **Tuned Random Forest + Hybrid_FS (১০টি feature)** model দিয়ে বানানো একটা
real-time prediction prototype। কোনো নতুন model train করা হয়নি। App শুধু আগে থেকে সংরক্ষিত (frozen) model
load করে এবং ব্যবহারকারীর দেওয়া ৬টি তথ্য (বয়স, ওজন, উচ্চতা, মাসিকের নিয়মিততা, Testosterone, Antral Follicle Count)
থেকে ১০টি feature নিজে তৈরি করে prediction দেয়।

## ফোল্ডার গঠন (সব ফাইল একই স্তরে)

```
pcos_streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── PCOS_Tuned_Random_Forest_Hybrid_FS.joblib   <-- Drive থেকে এনে এখানে রাখতে হবে
```

`.joblib` ফাইলটি (প্রায় 6.4 MB) Drive-এর `Code_Mam_Final_Pipeline/Step_24_Streamlit_Deployment/` ফোল্ডারে আছে।
Step 24 চালালে ফাইলটি সেখানে তৈরি হয়। Step 24 নিজেই যাচাই করে যে এটি Step 23-এ পরীক্ষিত model-এর হুবহু কপি (SHA-256 মেলানো)।

## Local-এ চালানো

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud-এ deploy

1. তিনটি ফাইল (app.py, requirements.txt, .joblib) GitHub repository-র root-এ তোলো।
2. share.streamlit.io-এ repository, branch `main`, main file `app.py` বেছে নাও।
3. Advanced settings-এ Python **3.11** বেছে নাও, তারপর Deploy।

## App যা করে

- বাংলা ও English দুই ভাষা।
- ইনপুট যাচাই: ফাঁকা, শূন্য, অসম্ভব মান বাতিল; training range-এর বাইরের মানে সতর্কতা।
- BMI নিজে হিসাব করে।
- ফলাফলে সম্ভাবনা (%), 0.35–0.65 সীমায় "অনিশ্চিত" চিহ্ন, এবং প্রতিটি prediction-এর real latency।

## সীমাবদ্ধতা

- এটা গবেষণার প্রোটোটাইপ, ডাক্তারি রোগনির্ণয়ের যন্ত্র নয়।
- Model শুধু একটি data সেটে (৪০০০ জন) train ও পরীক্ষিত। অন্য জনগোষ্ঠীতে কাজ করবে এমন দাবি করা যাবে না।
- Validation-এ সব ভালো model-এর মধ্যে পার্থক্য খুব ছোট (প্রায় ২টি record), তাই "সবচেয়ে ভালো model" কথাটি সতর্কভাবে বলতে হবে।
- Hosted link থাকলেও Input validation ন্যূনতম, production-মানের নয়।
