# المساعد الذكي للمعلومات (Wiseflow)

**[English](README_EN.md) | [日本語](README_JP.md) | [한국어](README_KR.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [简体中文](README.md)**

🚀 **استخدم الذكاء الاصطناعي لاستخراج المعلومات التي تهتم بها حقاً من مصادر متعددة يومياً!**

المشكلة ليست في نقص المعلومات، بل في كيفية تصفية الضوضاء من المعلومات الهائلة للكشف عن المعلومات القيمة

## 🔥🔥🔥 الإصدار 4.0 من Wiseflow متوفر الآن!

https://github.com/user-attachments/assets/2c52c010-6ae7-47f4-bc1c-5880c4bd76f3

(الخدمة عبر الإنترنت غير متوفرة حالياً بسبب أسباب تقنية، نحن نعمل على تسريع الترقية)

بعد انتظار دام ثلاثة أشهر، نحن سعداء بإطلاق الإصدار 4.0 من Wiseflow! هذا الإصدار يجلب هيكلية جديدة 4.x، مع دعم لمنصات التواصل الاجتماعي وميزات جديدة متعددة.

الإصدار 4.x يتضمن WIS Crawler (مبني على Crawl4ai و MediaCrawler و Nodriver)، ويوفر دعم لمنصات التواصل الاجتماعي بالإضافة إلى صفحات الويب. الإصدار 4.0 يدعم حالياً منصتي Weibo و Kuaishou، مع خطط لإضافة المزيد من المنصات مثل:
WeChat Official Accounts و Xiaohongshu و Douyin و Bilibili و Zhihu...

ميزات جديدة أخرى في هيكلية 4.x تشمل:

- هيكلية جديدة تستخدم المزامنة وخيوط المعالجة بشكل مختلط، مما يرفع كفاءة المعالجة (مع تقليل استهلاك الذاكرة)؛
- ورث قدرات dispatcher من Crawl4ai 0.6.3، مما يوفر إدارة ذاكرة أكثر دقة؛
- دمج عميق بين Pre-Process من الإصدار 3.9 و Markdown Generation من Crawl4ai، مما يمنع المعالجة المكررة؛
- تحسين دعم مصادر RSS؛
- تحسين هيكل ملفات المشروع، أكثر وضوحاً ومتوافقاً مع معايير مشاريع Python المعاصرة؛
- استخدام uv لإدارة التبعيات، وتحسين ملف requirement.txt؛
- تحسين سكريبتات التشغيل (مع دعم Windows)، مما يجعل "التشغيل بنقرة واحدة" حقيقة؛
- تحسين عملية التكوين والنشر، البرنامج الخلفي لم يعد يعتمد على خدمة pocketbase، لذلك لا حاجة لتوفير بيانات اعتماد pocketbase في ملف .env، ولا يوجد قيود على إصدار pocketbase.

## 🧐  'البحث العميق' مقابل 'البحث الواسع'

أطلقنا على منتج Wiseflow مصطلح "البحث الواسع"، وهذا مقابل "البحث العميق" الشائع حالياً.

"البحث العميق" يتضمن تخطيطاً ديناميكياً لمسار البحث بواسطة LLM لاستكشاف صفحات مختلفة وجمع معلومات كافية للإجابة على سؤال محدد أو إنتاج تقرير. لكن في بعض الأحيان، نحن لا نبحث عن إجابة لسؤال محدد ولا نحتاج إلى استكشاف عميق، بل نحتاج فقط إلى جمع معلومات واسعة (مثل جمع معلومات الصناعة، معلومات الخلفية، معلومات العملاء، إلخ). في هذه الحالات، يكون الاتساع أكثر أهمية. رغم أن "البحث العميق" يمكنه تحقيق هذه المهمة، إلا أنه مثل استخدام مدفع لقتل بعوضة - غير فعال ومكلف. Wiseflow هو الأداة المثالية لمثل هذه السيناريوهات.

## ✋ ما الذي يجعل Wiseflow مختلفاً عن برامج الزحف المدعومة بالذكاء الاصطناعي الأخرى؟

- قدرة على الحصول على المعلومات من جميع المنصات، بما في ذلك صفحات الويب ومنصات التواصل الاجتماعي (حالياً يدعم Weibo و Kuaishou) ومصادر RSS ومحركات البحث؛
- ليس مجرد جمع، بل تحليل وتصفية تلقائية، ويعمل بشكل جيد مع نموذج ذكاء اصطناعي بحجم 14b فقط؛
- مصمم للمستخدم العادي (وليس للمطورين)، جاهز للاستخدام "من الصندوق"؛
- استقرار وموثوقية عالية من خلال التطوير المستمر، وكفاءة معالجة متوازنة بين موارد النظام والسرعة؛
- (مستقبلاً) قدرة على اكتشاف "المعلومات الخفية" المخفية في المعلومات المجمعة من خلال وحدة insight

……… نتطلع إلى انضمام المطورين المهتمين إلينا لبناء المساعد الذكي للمعلومات للجميع!

## 🌟 البدء السريع

**ثلاث خطوات فقط للبدء!**

### 📋 تحميل كود المصدر وتثبيت uv و pocketbase

- لنظام MacOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

- لنظام Windows:

**يجب على مستخدمي Windows تحميل أداة git bash مسبقاً وتنفيذ الأوامر التالية في bash [رابط تحميل bash](https://git-scm.com/downloads/win)**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/TeamWiseFlow/wiseflow.git
```

يمكنك أيضاً استخدام install_pocketbase.sh (لنظام MacOS/Linux) أو install_pocketbase.ps1 (لنظام Windows) للتثبيت.

### 📥 تكوين ملف .env

في مجلد wiseflow (المجلد الرئيسي للمشروع)، قم بإنشاء ملف .env بناءً على env_sample وإدخال المعلومات المطلوبة.

الإصدار 4.x لا يتطلب من المستخدم توفير بيانات اعتماد pocketbase في ملف .env، ولا يحد من إصدار pocketbase. كما ألغينا مؤقتاً إعداد Secondary Model، لذلك تحتاج فقط إلى أربعة معلمات أساسية:

- LLM_API_KEY="" # مفتاح خدمة LLM (يمكن استخدام أي مزود خدمة يوفر واجهة برمجة تطبيقات بتنسيق OpenAI، لا حاجة للإعداد عند استخدام ollama محلياً)
- LLM_API_BASE="https://api.siliconflow.cn/v1" # عنوان واجهة خدمة LLM
- JINA_API_KEY="" # مفتاح خدمة محرك البحث (نوصي بـ Jina، يمكن حتى للمستخدمين الشخصيين التقديم دون تسجيل)
- PRIMARY_MODEL="Qwen3-14B" # نوصي بـ Qwen3-14B أو نموذج تفكير بنفس المستوى

### 🚀  ابدأ!

- لنظام MacOS/Linux:

```bash
cd wiseflow
uv venv # فقط يحتاج إلى التنفيذ في المرة الأولى
python -m playwright install --with-deps chromium # فقط يحتاج إلى التنفيذ في المرة الأولى
chmod +x run.sh # فقط يحتاج إلى التنفيذ في المرة الأولى
./run.sh
```

- لنظام Windows:

```bash
cd wiseflow
uv venv # فقط يحتاج إلى التنفيذ في المرة الأولى
python -m playwright install --with-deps chromium # فقط يحتاج إلى التنفيذ في المرة الأولى
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser # فقط يحتاج إلى التنفيذ في المرة الأولى
.\run.ps1
```

للحصول على دليل استخدام مفصل، يرجى الرجوع إلى [docs/manual/manual_ar.md](./docs/manual/manual_ar.md)

## 📚 كيفية استخدام البيانات المجمعة من Wiseflow في برنامجك الخاص

يتم تخزين جميع البيانات المجمعة في pocketbase مباشرة، لذلك يمكنك الوصول إلى البيانات مباشرة من خلال قاعدة بيانات pocketbase.

PocketBase كقاعدة بيانات خفيفة الوزن شائعة الاستخدام، يتوفر حالياً SDK بلغات Go/Javascript/Python وغيرها.

ستقوم الخدمة عبر الإنترنت قريباً بإطلاق sync api، لدعم مزامنة نتائج الزحف عبر الإنترنت محلياً، لبناء "قاعدة معرفة ديناميكية" وغيرها، ابقوا على اطلاع:

  - عنوان الخدمة عبر الإنترنت: https://www.aiqingbaoguan.com/ 
  - أمثلة استخدام API للخدمة عبر الإنترنت: https://github.com/TeamWiseFlow/wiseflow_plus


## 🛡️ الترخيص

هذا المشروع مفتوح المصدر بموجب [Apache2.0](LICENSE).

للتعاون التجاري، يرجى الاتصال بـ **البريد الإلكتروني: zm.zhao@foxmail.com**

- يرجى من العملاء التجاريين الاتصال بنا للتسجيل، النسخة المفتوحة المصدر مجانية للأبد.

## 📬 معلومات الاتصال

لأي أسئلة أو اقتراحات، يرجى ترك تعليق في [issue](https://github.com/TeamWiseFlow/wiseflow/issues).

## 🤝 هذا المشروع مبني على المشاريع المفتوحة المصدر التالية:

- Crawl4ai (Open-source LLM Friendly Web Crawler & Scraper) https://github.com/unclecode/crawl4ai
- MediaCrawler (xhs/dy/wb/ks/bilibili/zhihu crawler) https://github.com/NanmiCoder/MediaCrawler
- NoDriver (Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas...) https://github.com/ultrafunkamsterdam/nodriver
- Pocketbase (Open Source realtime backend in 1 file) https://github.com/pocketbase/pocketbase
- Feedparser (Parse feeds in Python) https://github.com/kurtmckee/feedparser

تأثر تطوير هذا المشروع بـ [GNE](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor) و [AutoCrawler](https://github.com/kingname/AutoCrawler) و [SeeAct](https://github.com/OSU-NLP-Group/SeeAct).

## الاقتباس

إذا استخدمت أو استشهدت بجزء أو كل من هذا المشروع في عملك، يرجى تضمين المعلومات التالية:

```
Author：Wiseflow Team
https://github.com/TeamWiseFlow/wiseflow
Licensed under Apache2.0
``` 