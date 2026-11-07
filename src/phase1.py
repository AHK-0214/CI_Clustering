"""
Full Phase 1 run: data understanding and predictions.

This module only looks at the raw data and does not run any clustering algorithm,
per the assignment's requirement ("do not look at clustering output yet").
"""
import os
import pandas as pd

from . import data as data_mod
from . import regions as regions_mod
from . import viz as viz_mod
from . import density as density_mod
from .config import OUTPUT_DIR, S_BASE


# ---------------------------------------------------------------------
# Question D - predictions P1 through P4 (must be recorded before seeing any
# clustering output). Note: the "prediction", "reason" and "rejection_criterion"
# values are written in Persian on purpose, since they are the actual answer text
# submitted for the assignment's report; only the dict keys (column labels) are English.
# ---------------------------------------------------------------------
PREDICTIONS = {
    "P1": {
        "prediction": (
            "ساختار کشیده‌ای که به‌احتمال زیاد بین چند خوشه K-Means (k=12) تقسیم می‌شود، "
            "پشته میان‌اقیانوسی اطلس (Mid-Atlantic Ridge) است؛ همچنین بخش‌هایی از کمربند آتش "
            "در امتداد سواحل غربی آمریکای جنوبی و شرق آسیا که طولی و کشیده هستند."
        ),
        "reason": (
            "K-Means خوشه‌ها را با مرز تصمیم Voronoi حول k مرکز می‌سازد که برای خوشه‌های "
            "کروی/فشرده مناسب است. یک ساختار باریک و طولانی (مثل یک خط گسل) اگر طولش از قطر "
            "معمول یک خوشه Voronoi بیشتر باشد، توسط چند مرکز مجاور هم پوشش داده می‌شود و بین "
            "چند خوشه تکه‌تکه می‌شود."
        ),
        "rejection_criterion": (
            "اگر پس از اجرا مشاهده شود که کل طول پشته میان‌اقیانوسی در یک یا دو خوشه قرار گرفته "
            "(و نه در سه خوشه یا بیشتر)، این پیش‌بینی رد می‌شود."
        ),
    },
    "P2": {
        "prediction": (
            "با افزایش eps از مقدار کوچک به بزرگ، درصد نقاط نویز به‌صورت یکنوا کاهش و اندازه "
            "بزرگ‌ترین خوشه به‌صورت یکنوا افزایش می‌یابد. در eps بزرگ، به‌دلیل پدیده زنجیره‌ای "
            "(chaining) بین نواحی با چگالی متفاوت، یک خوشه غول‌پیکر بیشتر نواحی پرتراکم را در "
            "خود ادغام می‌کند. انتظار نداریم یک eps واحد بتواند هم‌زمان نواحی پرتراکم (مثل R1) و "
            "کم‌تراکم (مثل R3) را به‌خوبی خوشه‌بندی کند، چون طبق تعریف DBSCAN یک eps سراسری است "
            "و تفاوت دستوری در چگالی بین نواحی (مشاهده‌شده در فاز ۱-ب) این مصالحه را سخت می‌کند."
        ),
        "reason": (
            "DBSCAN نقاط را بر اساس شمارش همسایگان در شعاع ثابت eps هسته‌ای تعریف می‌کند؛ "
            "افزایش eps شعاع همسایگی هر نقطه را بزرگ‌تر و در نتیجه احتمال رسیدن به آستانه "
            "min_samples را بیشتر می‌کند، و نیز خوشه‌های مجاور را به هم متصل می‌کند."
        ),
        "rejection_criterion": (
            "اگر در جدول فاز ۳-آ یک بازه از eps پیدا شود که در آن هم درصد نویز در R1 (پرتراکم) پایین "
            "بماند و هم R3 (کم‌تراکم) به‌طور معقول جدا از نویز باقی بماند، بخش دوم این پیش‌بینی رد می‌شود."
        ),
    },
    "P3": {
        "prediction": (
            "Single Link بیشتر مستعد تشکیل یک خوشه بسیار بزرگ و چند خوشه بسیار کوچک (ساختار "
            "'rich-get-richer') است، در حالی‌که Average Link خوشه‌های متوازن‌تری می‌سازد."
        ),
        "reason": (
            "در Single Link فاصله دو خوشه برابر کمینه فاصله بین هر عضو از دو خوشه است؛ این معیار "
            "نسبت به پدیده زنجیره‌ای بسیار حساس است: کافی است یک مسیر نازک از نقاط نزدیک به هم دو "
            "توده بزرگ را به هم وصل کند. Average Link از میانگین همه فاصله‌های دوبه‌دو استفاده می‌کند "
            "که نسبت به چنین پل‌های نازکی مقاوم‌تر است و خوشه‌های فشرده‌تر و متوازن‌تر تولید می‌کند."
        ),
        "rejection_criterion": (
            "اگر در فاز ۳-پ توزیع اندازه خوشه‌های Average Link نامتوازن‌تر از Single Link باشد "
            "(مثلاً واریانس اندازه خوشه بیشتر)، این پیش‌بینی رد می‌شود."
        ),
    },
    "P4": {
        "prediction": (
            "برای سناریوی الف (منطقه‌بندی)، K-Means را مناسب‌تر می‌دانیم چون هر نقطه را به دقیق یک "
            "خوشه اختصاص می‌دهد (پوشش ۱۰۰٪) و تعداد مناطق (k) را می‌توان مستقیماً در بازه ۱۰ تا ۲۰ "
            "تنظیم کرد. برای سناریوی ب (کشف ساختار)، DBSCAN را مناسب‌تر می‌دانیم چون تعداد خوشه‌ها "
            "را از قبل تحمیل نمی‌کند و امکان نویز گذاشتن نقاط دورافتاده را می‌دهد."
        ),
        "reason": (
            "فرض‌های K-Means (خوشه‌های کروی حول مرکز) با هدف سناریوی الف (مناطق قابل‌توصیف و پوشش "
            "کامل) سازگارتر است؛ فرض DBSCAN (پیوستگی چگالی) با هدف سناریوی ب (کشف کمربندها بدون "
            "تعیین k) سازگارتر است."
        ),
        "rejection_criterion": (
            "اگر در فاز ۴ معلوم شود DBSCAN با تنظیم مناسب eps پوشش نزدیک به کامل و مناطق به همان "
            "اندازه قابل‌توصیف برای سناریوی الف تولید می‌کند، یا K-Means در کشف کمربندهای غیرکروی "
            "سناریوی ب عملکرد قابل قبولی دارد، بخش مربوطه این پیش‌بینی رد می‌شود."
        ),
    },
}


# ---------------------------------------------------------------------
# Question C - explanation of why plain Euclidean distance on lat/lon is misleading.
# Kept in Persian on purpose: this is the actual answer text for the assignment's report.
# ---------------------------------------------------------------------
DISTANCE_EXPLANATION = """
محاسبه مستقیم فاصله اقلیدسی روی ستون‌های خام Latitude و Longitude گمراه‌کننده است
زیرا این دو کمیت زاویه‌اند، نه مختصات کارتزین روی یک صفحه مسطح:

۱) ناپیوستگی در ۱۸۰ درجه طول جغرافیایی:
   دو نقطه با طول جغرافیایی ۱۷۹ و ۱۷۹- درجه (مثلاً نزدیک خط تغییر تاریخ در اقیانوس آرام) تنها
   حدود ۲ درجه (~۲۲۲ کیلومتر) از هم فاصله واقعی دارند، اما فاصله اقلیدسی خام آن‌ها روی محور طول
   جغرافیایی برابر ۳۵۸ درجه محاسبه می‌شود؛ یعنی تقریباً بیشینه ممکن. این دو نقطه که در واقعیت
   همسایه‌اند، در این متریک نادرست بسیار دور از هم به نظر می‌رسند.

۲) تاب‌خوردگی نزدیک قطب‌ها:
   طول هر درجه طول جغرافیایی با cos(عرض جغرافیایی) کوچک می‌شود. نزدیک قطب، اختلاف زیاد در طول
   جغرافیایی می‌تواند فاصله واقعی بسیار کمی را نشان دهد، درحالی‌که فاصله اقلیدسی خام آن را بزرگ
   محاسبه می‌کند.

نمایش سه‌بعدی روی کره واحد (x=cos φ cos λ, y=cos φ sin λ, z=sin φ) هر دو مشکل را برطرف می‌کند:
این تبدیل یک نگاشت پیوسته و بدون درز (seamless) از سطح کره به فضای اقلیدسی سه‌بعدی است، بنابراین
نقاط نزدیک جغرافیایی (از جمله دو طرف نصف‌النهار ۱۸۰ درجه) در این فضا نیز به هم نزدیک‌اند، و فاصله
اقلیدسی در این فضا (پس از تبدیل به فاصله سطحی با فرمول کمانی) با فاصله واقعی روی کره سازگار است.
""".strip()


def run_phase1(save_outputs: bool = True) -> dict:
    """Run all of Phase 1 and return a results dict (for use in the notebook)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load and clean ----
    df = data_mod.load_clean_earthquakes()
    n_total = len(df)

    # ---- Question A: map and three regions ----
    fig_map, _ = viz_mod.plot_world_map(
        df, save_path=os.path.join(OUTPUT_DIR, "01_world_map.png") if save_outputs else None
    )
    fig_map_geo = viz_mod.plot_world_map_geo(
        df, save_path=os.path.join(OUTPUT_DIR, "01b_world_map_geo.html") if save_outputs else None
    )
    fig_regions, _ = viz_mod.plot_world_map_with_regions(
        df, save_path=os.path.join(OUTPUT_DIR, "02_world_map_regions.png") if save_outputs else None
    )
    region_table = regions_mod.region_summary_table(df)

    # ---- Question B: density (distance to 10th neighbor) ----
    dist_km = density_mod.kth_neighbor_distance_km(df)
    fig_hist, _ = viz_mod.plot_density_histogram(
        dist_km, save_path=os.path.join(OUTPUT_DIR, "03_density_histogram.png") if save_outputs else None
    )
    density_table = density_mod.region_density_table(df, dist_km, regions_mod)

    if save_outputs:
        region_table.to_csv(os.path.join(OUTPUT_DIR, "region_table.csv"), index=False)
        density_table.to_csv(os.path.join(OUTPUT_DIR, "density_table.csv"), index=False)
        df.assign(dist_to_10th_neighbor_km=dist_km).to_csv(
            os.path.join(OUTPUT_DIR, "cleaned_earthquakes_with_density.csv"), index=False
        )
        with open(os.path.join(OUTPUT_DIR, "distance_explanation.txt"), "w", encoding="utf-8") as f:
            f.write(DISTANCE_EXPLANATION)
        predictions_df = pd.DataFrame(PREDICTIONS).T
        predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_P1_P4.csv"), encoding="utf-8-sig")

    return {
        "df": df,
        "n_total": n_total,
        "seed_base": S_BASE,
        "region_table": region_table,
        "density_table": density_table,
        "dist_km": dist_km,
        "figures": {"map": fig_map, "map_geo": fig_map_geo, "map_regions": fig_regions, "hist": fig_hist},
        "distance_explanation": DISTANCE_EXPLANATION,
        "predictions": PREDICTIONS,
    }


if __name__ == "__main__":
    results = run_phase1()
    print(f"Total number of earthquakes after cleaning: {results['n_total']}")
    print(results["region_table"])
    print(results["density_table"])
