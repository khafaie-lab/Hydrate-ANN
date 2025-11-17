# Hydrate-ANN
The full dataset and the source code used to generate the results in this study will be made publicly available here following the successful defense of the doctoral dissertation and the peer-reviewed publication of the primary manuscript. This is to ensure the integrity of the academic review process. Thank you for your understanding.
==================================================

مجموعه کامل داده‌ها و کد منبع استفاده شده برای تولید نتایج در این مطالعه، پس از دفاع موفقیت‌آمیز رساله دکتری و انتشار مقاله اصلی در یک مجله داوری شده، در اینجا به صورت عمومی در دسترس قرار خواهد گرفت. این کار برای تضمین یکپارچگی فرآیند بررسی آکادمیک انجام می‌شود. از درک شما سپاسگزاریم.
==================================================

فایل اکسل بالا، ضمیمه پایان نامه دکترا مهندسی شیمی موسی خفائی میباشد. این پیوست، مقادیر کامل ماتریس‌های وزن و بردارهای بایاس را برای مدل نهایی شبکه عصبی پرسپترون چندلایه (MLP)، که در فصل پنجم پایان نامه عملکرد آن ارزیابی شد، ارائه می‌دهد. ارائه این پارامترها با هدف تضمین تکرارپذیری کامل (full reproducibility) این پژوهش صورت گرفته است. با استفاده از این مقادیر، هر محققی می‌تواند مدل توسعه‌یافته را دقیقاً بازسازی کرده و از آن برای پیش‌بینی‌های آتی استفاده نماید. معماری شبکه نهایی به شرح زیر بوده است:
نوع شبکه: پرسپترون چندلایه پیش‌خور (Feedforward MLP)
معماری: [تعداد ورودی‌ها] - 9 - 9 - 9 - 9 - 9 - 1
توابع فعال‌سازی: tansig برای 5 لایه پنهان و purelin برای لایه خروجی
الگوریتم آموزش: لونبرگ-مارکوارت (Levenberg-Marquardt)
تمام پارامترهای زیر برای داده‌های نرمال‌شده (normalized) در بازه [-1, 1] ارائه شده‌اند. برای استفاده از مدل، داده‌های ورودی جدید ابتدا باید با استفاده از مقادیر حداقل و حداکثر گزارش شده در فصل چهارم نرمال‌سازی شوند و خروجی نهایی مدل نیز باید به مقیاس واقعی دی-نرمال شود.
==================================================

Accurate prediction of semi-clathrate gas hydrate formation conditions is of particular importance for novel industrial applications such as gas storage and CO2 capture, owing to their stability under mild thermodynamic conditions. However, traditional thermodynamic models face limitations in predicting the behavior of these multi-component systems. This research aimed to develop and validate a comprehensive and reliable Artificial Neural Network (ANN) model for predicting the equilibrium formation temperature of semi-clathrate hydrates in systems of (water + TBAB/TBAC salts + CH4/CO2/N2 gas mixtures).
To this end, an extensive database comprising 1841 equilibrium data points was first compiled from reputable scientific literature. Subsequently, for industrial validation, approximately 70 new data points were experimentally measured for a real natural gas composition from an Iranian gas field in the presence of various promoter concentrations. An optimized Multi-Layer Perceptron (MLP) neural network (5 hidden layers, 9 neurons each) was trained on the literature-based data, and its performance was compared with a hybrid Particle Swarm Optimization (PSO-ANN) model. Finally, the optimal model was evaluated in an external validation test using the new and previously unseen industrial experimental data.
The results demonstrated that the base ANN model could predict the hydrate formation temperature on the literature-based test set with very high accuracy (R² = 0.9955, RMSE = 0.024 K), exhibiting better generalization performance than the PSO-ANN model (R² = 0.9747). More importantly, the ANN model also succeeded in the external validation stage, predicting the equilibrium conditions of the industrial natural gas with excellent and acceptable accuracy (R² > 0.989), which proves its credibility and reliability for engineering applications. Furthermore, it was determined that the model achieves this level of accuracy using only fundamental inputs (pressure and composition), obviating the need for additional thermophysical parameters. This research provides a rapid and accurate engineering tool for predicting the behavior of semi-clathrate hydrates and highlights the importance of a two-stage validation strategy for developing robust data-driven models.




