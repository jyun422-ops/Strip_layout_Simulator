import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Circle, Rectangle
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import tempfile
import os
import glob
import hashlib
import io
import math
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# [-1] matplotlib 한글 폰트 자동 설정
# ============================================================
def setup_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                  "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Batang"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams['font.family'] = name
            mpl.rcParams['axes.unicode_minus'] = False
            return name

    search_paths = [
        "/usr/share/fonts/**/*Nanum*.ttf", "/usr/share/fonts/**/*Nanum*.otf",
        "/usr/share/fonts/**/*Noto*CJK*.ttc", "/usr/share/fonts/**/*Noto*CJK*.otf",
        "/usr/share/fonts/**/*malgun*.ttf", "/Library/Fonts/**/*Nanum*.ttf",
        "./fonts/*.ttf", "./fonts/*.otf",
    ]
    for pattern in search_paths:
        for fpath in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(fpath)
                font_name = fm.FontProperties(fname=fpath).get_name()
                mpl.rcParams['font.family'] = font_name
                mpl.rcParams['axes.unicode_minus'] = False
                return font_name
            except Exception:
                continue

    mpl.rcParams['axes.unicode_minus'] = False
    return None

KOREAN_FONT_FOUND = setup_korean_font()

# ============================================================
# [0] DXF 단위 변환 테이블
# ============================================================
INSUNITS_TO_MM = {
    0: None, 1: 25.4, 2: 304.8, 3: 1609344.0, 4: 1.0, 
    5: 10.0, 6: 1000.0, 13: 100.0,
}

def get_unit_factor(doc):
    try:
        code = doc.header.get('$INSUNITS', 0)
    except Exception:
        code = 0
    factor = INSUNITS_TO_MM.get(code, None)
    if factor is None:
        return 1.0, "⚠️ 도면에 단위 정보가 없거나 인식할 수 없어 **mm으로 간주**하고 계산합니다."
    if factor == 1.0:
        return 1.0, "✅ 도면 단위: mm (변환 불필요)"
    return factor, f"✅ 도면 단위 자동 인식: 환산 계수 ×{factor} 적용하여 mm로 변환했습니다."


# ============================================================
# [1] 핵심 알고리즘 (초정밀 파고들기 스캔 적용)
# ============================================================
def calculate_1d_pitch(geom, bridge):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    buffered_geom = geom.buffer(bridge, resolution=4)
    dx = w + bridge
    step = w / 30

    while dx > 0:
        test_geom = translate(geom, xoff=dx, yoff=0)
        if buffered_geom.intersects(test_geom):
            dx += step
            break
        dx -= step

    if dx <= 0: dx = step
    fine_step = step / 10
    while dx > 0:
        test_geom = translate(geom, xoff=dx, yoff=0)
        if buffered_geom.intersects(test_geom):
            dx += fine_step
            break
        dx -= fine_step
    return dx

def find_nesting_offset(part_a, part_b, p_base, bridge):
    buffered_a = part_a.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part_a.bounds
    w, h = maxx - minx, maxy - miny

    reps = int(np.ceil((maxx - minx + p_base) / p_base)) + 2
    base_geoms = [translate(buffered_a, xoff=i*p_base, yoff=0) for i in range(-2, reps+1)]
    base_row = unary_union(base_geoms)

    best_dx, best_dy = 0, float('inf')
    best_part_b = None

    for dx in np.linspace(0, p_base, 90):
        dy = h + bridge * 2  
        step = max(0.5, min(h / 20, bridge / 2)) 
        test_b = translate(part_b, xoff=dx, yoff=dy)

        while not base_row.intersects(test_b):
            dy -= step
            if dy < -h * 1.5: break 
            test_b = translate(part_b, xoff=dx, yoff=dy)

        dy += step 
        fine_step = step / 10
        test_b = translate(part_b, xoff=dx, yoff=dy)

        while not base_row.intersects(test_b):
            dy -= fine_step
            if dy < -h * 1.5: break
            test_b = translate(part_b, xoff=dx, yoff=dy)
        dy += fine_step 

        if dy < best_dy:
            best_dy = dy
            best_dx = dx
            best_part_b = translate(part_b, xoff=dx, yoff=dy)

    if best_part_b:
        return best_dx, best_dy, unary_union([part_a, best_part_b]), best_part_b
    return None, None, None, None

def check_multi_row_clash(parts, bridge):
    """3열 이상 배열 시 간섭이 발생하는지 기하학적으로 검증합니다."""
    if len(parts) < 3: return False
    buf0 = parts[0].buffer(bridge - 0.05, resolution=4)
    for r in range(2, len(parts)):
        if buf0.intersects(parts[r]): return True
    return False

# ============================================================
# [2] DXF 읽기 및 시각화(Rendering) (블록 자동 분해 기능 포함)
# ============================================================
def extract_entities_recursive(entity_container):
    target_types = {'LWPOLYLINE', 'POLYLINE', 'CIRCLE', 'ELLIPSE', 'SPLINE'}
    extracted = []
    
    for entity in entity_container:
        if entity.dxftype() == 'INSERT':
            try:
                for virt_entity in entity.virtual_entities():
                    if virt_entity.dxftype() == 'INSERT':
                        extracted.extend(extract_entities_recursive([virt_entity]))
                    elif virt_entity.dxftype() in target_types:
                        extracted.append(virt_entity)
            except Exception:
                continue
        elif entity.dxftype() in target_types:
            extracted.append(entity)
            
    return extracted

def read_part_with_holes(msp, unit_factor):
    candidates = []
    flatten_tol = 0.1 if unit_factor == 0 else max(1e-4, 0.05 / unit_factor)
    
    valid_entities = extract_entities_recursive(msp)
    
    for entity in valid_entities:
        try:
            dxftype = entity.dxftype()
            p = path.make_path(entity)
            coords = [(v.x * unit_factor, v.y * unit_factor) for v in p.flattening(distance=flatten_tol)]
            if len(coords) < 3: continue

            poly = Polygon(coords).buffer(0)
            
            if poly.is_empty or poly.area < 1e-4: 
                continue
                
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda a: a.area)
            candidates.append(poly)
        except Exception:
            continue

    if not candidates: return None, []

    # 원형도(Circularity) 필터링 - 참조 원(치수, 피치원) 오인 방지
    valid_outer_candidates = []
    pure_circles = []

    for poly in candidates:
        area = poly.area
        perimeter = poly.length
        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if circularity > 0.95:  
            pure_circles.append(poly)
        else:
            valid_outer_candidates.append(poly)

    if valid_outer_candidates:
        outer = max(valid_outer_candidates, key=lambda a: a.area)
    else:
        outer = max(pure_circles, key=lambda a: a.area)

    holes = [c for c in candidates if c is not outer and outer.contains(c.buffer(-1e-6))]
    
    net_part = Polygon(outer.exterior.coords, [h.exterior.coords for h in holes]) if holes else outer
    return net_part, holes


def plot_polygon(ax, poly, color, lw=1.5, alpha=0.5):
    verts = []
    codes = []
    
    ext_coords = list(poly.exterior.coords)
    if not poly.exterior.is_ccw:
        ext_coords = ext_coords[::-1]
        
    verts.extend(ext_coords)
    codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext_coords) - 2) + [MplPath.CLOSEPOLY]
    
    for interior in poly.interiors:
        int_coords = list(interior.coords)
        if interior.is_ccw:
            int_coords = int_coords[::-1]
        verts.extend(int_coords)
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(int_coords) - 2) + [MplPath.CLOSEPOLY]
        
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor=color, linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    
    for interior in poly.interiors:
        xs, ys = interior.xy
        ax.fill(xs, ys, color='white', alpha=1.0, zorder=3) 
        ax.plot(xs, ys, color=color, linewidth=lw * 0.8, zorder=4) 
        
    ax.plot(*poly.exterior.xy, color=color, linewidth=lw, zorder=4)

# ============================================================
# [3] UI 렌더링 헬퍼 함수
# ============================================================
def render_case_column(col, label, results, best, used_fallback, colors, margin, carrier_width):
    with col:
        st.subheader(f"{label} ({best['angle']}°)")
        st.caption(f"이용율: :blue[**{best['util']:.2f}%**] | 단가: :blue[**{int(best['cost']):,}원**]")
        if used_fallback:
            st.warning("⚠️ 압연방향 제약을 만족하는 각도가 없어 제약을 무시한 최적값입니다.")

        fig, ax = plt.subplots(figsize=(6, 6))
        for geom, color in zip(best['parts'], colors):
            plot_polygon(ax, geom, color)
        
        all_geom = unary_union(best['parts'])
        sx1, sy1 = all_geom.bounds[0], all_geom.bounds[1] - margin - carrier_width
        sy2 = all_geom.bounds[3] + margin + carrier_width
        
        ax.plot([sx1, sx1 + best['p'], sx1 + best['p'], sx1, sx1], [sy1, sy1, sy2, sy2, sy1], 
                color='red', linestyle='--', linewidth=2.5,
                label=f"1피치 소요 면적\n(폭: {best['w']:.1f} × 피치: {best['p']:.2f})")
        
        ax.axis('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=9, framealpha=1.0)
        fig.subplots_adjust(bottom=0.28)  
        st.pyplot(fig)

        df = pd.DataFrame(results)  
        df['is_chosen'] = df['각도'] == f"{best['angle']}°"
        df = df.sort_values(by=['is_chosen', '소재이용율(%)'], ascending=[False, False]).reset_index(drop=True)
        df = df.drop(columns=['is_chosen'])
        
        def highlight(row):
            is_chosen = (row['각도'] == f"{best['angle']}°")
            if is_chosen: 
                return ['color: blue; font-weight: bold; background-color: #e6f2ff;'] * len(row)
            if row['압연방향 적합'] == 'X': 
                return ['background-color: #ffe6e6;'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.style.apply(highlight, axis=1).format({'피치(mm)': '{:.2f}', '소재폭(mm)': '{:.2f}', '소재이용율(%)': '{:.2f}'}), use_container_width=True)


# ============================================================
# [4] 데이터 추출 헬퍼 함수 (DXF / Excel / PDF)
# ============================================================
def plot_strip_layout(parts_and_colors, pitch, part_zone_width, margin, carrier_width, pilot_dia, total_stations):
    strip_width = part_zone_width + carrier_width * 2
    all_geoms = unary_union([p[0] for p in parts_and_colors])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    total_length = (pitch * (total_stations - 1)) + part_length + (pitch * 0.4)

    fig, ax = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    ax.plot([0, total_length, total_length, 0, 0], [0, 0, strip_width, strip_width, 0],
            color='red', linestyle='-', linewidth=2.5,
            label=f'금형 코어 최소 사이즈\n(가로: {total_length:.1f} x 세로: {strip_width:.1f})')

    if carrier_width > 0:
        ax.add_patch(Rectangle((0, 0), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))
        ax.add_patch(Rectangle((0, strip_width - carrier_width), total_length, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', label='캐리어(스켈레톤) 영역', zorder=1))

    y_offset, x_offset = -miny + margin + carrier_width, -minx + (pitch * 0.2)

    for i in range(total_stations):
        for geom, color in parts_and_colors:
            shifted = translate(geom, xoff=x_offset + (i * pitch), yoff=y_offset)
            plot_polygon(ax, shifted, color, lw=1.5, alpha=0.5)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax.add_patch(Circle((pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax.plot([pitch * (i + 1), pitch * (i + 1)], [0, strip_width], color='black', linestyle=':', alpha=0.4, zorder=1)

    ax.axis('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig

def render_strip_section(label, best, total_stations, margin, carrier_width, pilot_dia, colors):
    all_geoms = unary_union(best['parts'])
    minx, miny, maxx, maxy = all_geoms.bounds
    part_length = maxx - minx
    l_val = (best['p'] * (total_stations - 1)) + part_length + (best['p'] * 0.4)
    
    st.info(f"📐 **{label} 금형 코어 최소 사이즈:** 가로(L) :blue[**{l_val:.1f} mm**] × 세로(W) :blue[**{best['w']:.1f} mm**] (캐리어 {carrier_width}mm 포함)  |  피치(P) :blue[**{best['p']:.2f} mm**] × **{total_stations}**스테이션")
    part_zone = best['w'] - carrier_width * 2
    fig = plot_strip_layout(list(zip(best['parts'], colors)), best['p'], part_zone, margin, carrier_width, pilot_dia, total_stations)
    st.pyplot(fig)

def generate_dxf_bytes(tuned_parts, tune_pitch, tune_width, total_stations, margin, carrier_width, pilot_dia, x_shift, y_shift):
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 4  
    msp = doc.modelspace()
    doc.layers.add("STRIP_EDGE", color=1)
    doc.layers.add("PARTS", color=7)
    doc.layers.add("PILOT_HOLES", color=5)

    all_geom_tuned = unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)

    msp.add_lwpolyline([(0, 0), (total_length_tuned, 0)], dxfattribs={'layer': 'STRIP_EDGE'})
    msp.add_lwpolyline([(0, tune_width), (total_length_tuned, tune_width)], dxfattribs={'layer': 'STRIP_EDGE'})
    
    def add_poly_to_block(poly, layer_name, target_block):
        if poly.geom_type == 'MultiPolygon':
            for p in poly.geoms: add_poly_to_block(p, layer_name, target_block)
            return
        target_block.add_lwpolyline(list(poly.exterior.coords), dxfattribs={'layer': layer_name, 'closed': True})
        for interior in poly.interiors:
            target_block.add_lwpolyline(list(interior.coords), dxfattribs={'layer': layer_name, 'closed': True})

    part_centroids = []
    for idx, geom in enumerate(tuned_parts):
        block_name = f"PART_BLOCK_{idx + 1}"
        if block_name not in doc.blocks:
            block = doc.blocks.new(name=block_name)
            cx, cy = geom.centroid.x, geom.centroid.y
            centered_geom = translate(geom, xoff=-cx, yoff=-cy)
            add_poly_to_block(centered_geom, "PARTS", block)
            part_centroids.append((cx, cy))

    for i in range(total_stations):
        station_x = x_shift + (i * tune_pitch)
        station_y = y_shift
        
        for idx in range(len(tuned_parts)):
            block_name = f"PART_BLOCK_{idx + 1}"
            cx, cy = part_centroids[idx]
            msp.add_blockref(block_name, insert=(station_x + cx, station_y + cy))

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            cx_p, cy_p = tune_pitch * (i + 0.5), carrier_width / 2
            msp.add_circle(center=(cx_p, cy_p), radius=pilot_dia / 2, dxfattribs={'layer': 'PILOT_HOLES'})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        doc.saveas(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f: dxf_bytes = f.read()
    os.remove(tmp_path)
    return dxf_bytes

def generate_excel_report(best_method_name, tune_pitch, tune_width, tune_util, tune_cost, total_stations, mat_type, material_thickness, material_price, material_density, s_res, i_res, z_res):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df = pd.DataFrame({
            "항목": ["채택된 배열 방식", "소재 분류", "소재 두께 (t)", "소재 비중", "단가 (원/kg)", "총 스테이션 수", "최종 피치 (mm)", "최종 소재 폭 (mm)", "최종 소재이용율 (%)", "최종 1개당 단가 (원)"],
            "값": [best_method_name, mat_type, material_thickness, material_density, material_price, total_stations, tune_pitch, tune_width, f"{tune_util:.2f}", int(tune_cost)]
        })
        summary_df.to_excel(writer, sheet_name="최종 요약", index=False)
        
        all_dfs = []
        if s_res: 
            df_s = pd.DataFrame(s_res)
            df_s.insert(0, '배열 방식', '단일 배열')
            all_dfs.append(df_s)
        if i_res: 
            df_i = pd.DataFrame(i_res)
            df_i.insert(0, '배열 방식', '180도 교차 배열')
            all_dfs.append(df_i)
        if z_res: 
            df_z = pd.DataFrame(z_res)
            df_z.insert(0, '배열 방식', '지그재그 배열')
            all_dfs.append(df_z)
            
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_excel(writer, sheet_name="전체 시뮬레이션 데이터", index=False)
            
    return output.getvalue()

def generate_pdf_report(fig_layout, best_method_name, tune_pitch, tune_width, tune_util, tune_cost, total_stations, mat_type, material_thickness):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        fig_text, ax_text = plt.subplots(figsize=(8, 5))
        ax_text.axis('off')
        
        title_font = {'fontsize': 16, 'fontweight': 'bold'}
        body_font = {'fontsize': 12}
        
        summary_text = (
            f"■ 소재 정보\n"
            f" - 분류: {mat_type}\n"
            f" - 두께: {material_thickness} t\n\n"
            f"■ 레이아웃 설계 결과\n"
            f" - 채택 배열: {best_method_name}\n"
            f" - 총 공정수: {total_stations} 스테이션\n"
            f" - 결정 피치: {tune_pitch:.2f} mm\n"
            f" - 결정 폭: {tune_width:.2f} mm\n\n"
            f"■ 최종 산출물\n"
            f" - 소재이용율: {tune_util:.2f} %\n"
            f" - 1개당 예상 단가: {int(tune_cost):,} 원\n"
        )
        
        ax_text.text(0.05, 0.95, "프레스 레이아웃 시뮬레이션 리포트", fontdict=title_font, va='top')
        ax_text.text(0.05, 0.75, summary_text, fontdict=body_font, va='top')
        
        pdf.savefig(fig_text)
        plt.close(fig_text)
        pdf.savefig(fig_layout)
        
    return pdf_buf.getvalue()


# ============================================================
# [5] 웹사이트 화면 및 사이드바
# ============================================================
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프로그레시브 금형 스트립 설계 시뮬레이터")

st.sidebar.header("📝 1. 소재 조건 입력")
mat_type = st.sidebar.radio("소재 특성 분류", ["일반 철강/연질 (SPCC, AL 등)", "고장력강/경질 (STS, SUS 등)"])
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)

if "일반" in mat_type:
    rec_bridge, rec_margin = max(1.2, 1.2 * material_thickness), max(1.5, 1.5 * material_thickness)
else:
    rec_bridge, rec_margin = max(1.5, 1.5 * material_thickness), max(2.0, 1.8 * material_thickness)

st.sidebar.header("📏 2. 배열 간격 (다이 강도 고려)")
bridge = st.sidebar.number_input("부품간 최소 간격 (mm)", value=float(round(rec_bridge, 1)), step=0.1)
margin = st.sidebar.number_input("가장자리 마진 (mm)", value=float(round(rec_margin, 1)), step=0.1)

st.sidebar.header("🔧 3. 캐리어 & 파일럿 설계")
carrier_width = st.sidebar.number_input("캐리어(스켈레톤) 폭 (mm)", value=float(round(max(4.0, 3 * material_thickness), 1)), step=0.5, help="부품을 다음 스테이션으로 이송시키는 스켈레톤 밴드 폭. 스트립 상/하단에 배치됩니다.")
pilot_dia = st.sidebar.number_input("파일럿 홀 지름 (mm)", value=4.0, step=0.5)
if not (carrier_width > 0 and 0 < pilot_dia < carrier_width):
    st.sidebar.caption("⚠️ 파일럿 홀 지름이 캐리어 폭보다 크거나 같아 Layout도에 표시되지 않습니다.")

st.sidebar.header("🛠️ 4. Layout도 설계")
st_notch = st.sidebar.number_input("노칭 / 파일럿 홀", value=1, step=1)
st_pierce = st.sidebar.number_input("피어싱 (내측 홀 타발)", value=1, step=1)
st_bend = st.sidebar.number_input("벤딩", value=0, step=1)
st_form = st.sidebar.number_input("포밍", value=0, step=1)
st_final_notch = st.sidebar.number_input("노칭 (최종 낙하)", value=1, step=1)
st_idle = st.sidebar.number_input("아이들 피치 (빈 구간)", value=1, step=1)
st_simul = st.sidebar.number_input("➖ 동시 성형 (중복 차감)", value=0, step=1)

total_stations = max(1, int((st_notch + st_pierce + st_bend + st_form + st_final_notch + st_idle) - st_simul))
st.sidebar.info(f"**총 예상 스테이션: {total_stations} 피치**")

st.sidebar.header("🧭 5. 압연방향(그레인) 제약")
apply_rolling_constraint = st.sidebar.checkbox("벤딩 라인 - 압연방향 최소 이격각 적용", value=(st_bend > 0), help="프레스 피드 방향(=압연방향, 도면 X축과 평행)과 벤딩 라인이 너무 나란하면 성형 시 크랙 위험이 커집니다.")
bend_angles_input = st.sidebar.text_input("벤딩 라인 각도 (°, 쉼표로 다중 입력)", value="0", disabled=not apply_rolling_constraint)
min_angle_from_rolling = st.sidebar.number_input("최소 이격각 (°, 통상 30~45° 권장)", value=30.0, step=5.0, disabled=not apply_rolling_constraint)

bend_line_angles = []
if apply_rolling_constraint:
    try:
        bend_line_angles = [float(x.strip()) for x in bend_angles_input.split(',')]
    except ValueError:
        st.sidebar.error("❌ 벤딩 라인 각도는 숫자와 쉼표(,)로만 입력해주세요. (예: 0, 90)")
        st.stop()

# ⭐ 추가: 3열 이상 다열 배열 설정 기능
st.sidebar.header("⚙️ 6. 다열 배열 설정")
num_rows = st.sidebar.number_input("교차/지그재그 배열 열(Row) 수", min_value=2, max_value=10, value=2, step=1, help="2열 이상의 다열 배열 시뮬레이션에 적용됩니다. 단일 배열은 항상 1열 기준으로 고정 계산됩니다.")

# ============================================================
# [6] 메인 화면 동작 및 결과 캐싱 로직
# ============================================================
st.markdown("#### 📂 1. 도면 업로드")
st.info("💡 **DXF 업로드 시 주의사항:** 도면의 부품 외곽선과 피어싱 홀은 **닫힌 폴리라인**이나 **블록(BLOCK)** 형태로 묶여 있어야 정확히 인식됩니다.")
uploaded_file = st.file_uploader("DXF 전개도면을 업로드하세요.", type=['dxf'])

st.markdown("#### 📐 2. 계산 각도 옵션 선택")
angle_step_option = st.radio("배열 회전 탐색 각도 간격을 선택하세요:", ["10도씩 (기본, 빠른 계산)", "5도씩 (정밀 계산)"], horizontal=True)
angle_step = 5 if "5도" in angle_step_option else 10

file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() if uploaded_file else None

current_params = (
    file_hash, bridge, margin, carrier_width,
    material_thickness, material_density, material_price, 
    apply_rolling_constraint, tuple(bend_line_angles), min_angle_from_rolling, angle_step, num_rows
)

if 'last_params' not in st.session_state:
    st.session_state.last_params = None

recalculated = False

if uploaded_file is not None:
    if st.session_state.last_params != current_params:
        recalculated = True 
        with st.spinner('안전 간격 적용 및 다열(Multi-row) 파고들기를 분석 중입니다...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            doc = ezdxf.readfile(tmp_path)
            msp = doc.modelspace()
            os.remove(tmp_path)

            unit_factor, unit_msg = get_unit_factor(doc)
            part, holes = read_part_with_holes(msp, unit_factor)
            
            if part is None:
                st.error("❌ 도면에서 다각형 기하학 정보를 찾을 수 없습니다. CAD 도면을 확인해주세요.")
                st.stop()
                
            if part.geom_type == 'MultiPolygon':
                part = max(part.geoms, key=lambda a: a.area)
                
            part_area, pair_area = part.area, part.area * 2

            single_results, inter_results, zigzag_results = [], [], []
            best_s = best_i = best_z = None
            fallback_s = fallback_i = fallback_z = None

            for angle in range(0, 180, angle_step):
                rotated_part = rotate(part, angle, origin='centroid')
                p_base = calculate_1d_pitch(rotated_part, bridge)

                valid = True
                if apply_rolling_constraint:
                    for b_angle in bend_line_angles:
                        eff_angle = (b_angle + angle) % 180
                        dist_from_parallel = min(eff_angle, 180 - eff_angle)
                        if dist_from_parallel < min_angle_from_rolling:
                            valid = False; break

                # [1] 단일 배열 계산 (항상 1열 기준)
                minx, miny, maxx, maxy = rotated_part.bounds
                w_s = (maxy - miny) + margin * 2 + carrier_width * 2
                util_s = (part_area / (p_base * w_s)) * 100
                cost_s = (((p_base * w_s * material_thickness) * material_density) / 1_000_000) * material_price
                record_s = {'util': util_s, 'cost': cost_s, 'angle': angle, 'parts': [rotated_part], 'w': w_s, 'p': p_base}
                single_results.append({
                    '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_s, 2),
                    '소재이용율(%)': round(util_s, 2), '1개당 원가(원)': int(cost_s), '압연방향 적합': 'O' if valid else 'X'
                })
                if fallback_s is None or util_s > fallback_s['util']: fallback_s = record_s
                if valid and (best_s is None or util_s > best_s['util']): best_s = record_s

                # [2] 교차 배열 계산 (다열 확장 적용)
                part_180 = rotate(rotated_part, 180, origin='centroid')
                dx_i1, dy_i1, _, _ = find_nesting_offset(rotated_part, part_180, p_base, bridge)
                if dx_i1 is not None:
                    dx_i2, dy_i2, _, _ = find_nesting_offset(part_180, rotated_part, p_base, bridge)
                    if dx_i2 is not None:
                        i_parts = [rotated_part]
                        cx, cy = 0.0, 0.0
                        for r in range(1, num_rows):
                            if r % 2 == 1:
                                cx += dx_i1; cy += dy_i1
                                i_parts.append(translate(part_180, xoff=cx, yoff=cy))
                            else:
                                cx += dx_i2; cy += dy_i2
                                i_parts.append(translate(rotated_part, xoff=cx, yoff=cy))
                        
                        if not check_multi_row_clash(i_parts, bridge):
                            geom_i = unary_union(i_parts)
                            w_i = (geom_i.bounds[3] - geom_i.bounds[1]) + margin * 2 + carrier_width * 2
                            util_i = (part_area * num_rows / (p_base * w_i)) * 100
                            cost_i = (((p_base * w_i * material_thickness) * material_density) / 1_000_000) * material_price / num_rows
                            record_i = {'util': util_i, 'cost': cost_i, 'angle': angle, 'parts': i_parts, 'w': w_i, 'p': p_base}
                            inter_results.append({
                                '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_i, 2),
                                '소재이용율(%)': round(util_i, 2), '1개당 원가(원)': int(cost_i), '압연방향 적합': 'O' if valid else 'X'
                            })
                            if fallback_i is None or util_i > fallback_i['util']: fallback_i = record_i
                            if valid and (best_i is None or util_i > best_i['util']): best_i = record_i

                # [3] 지그재그 배열 계산 (다열 확장 적용)
                dx_z, dy_z, _, _ = find_nesting_offset(rotated_part, rotated_part, p_base, bridge)
                if dx_z is not None:
                    z_parts = [rotated_part]
                    cx, cy = 0.0, 0.0
                    for r in range(1, num_rows):
                        cx += dx_z; cy += dy_z
                        z_parts.append(translate(rotated_part, xoff=cx, yoff=cy))
                        
                    if not check_multi_row_clash(z_parts, bridge):
                        geom_z = unary_union(z_parts)
                        w_z = (geom_z.bounds[3] - geom_z.bounds[1]) + margin * 2 + carrier_width * 2
                        util_z = (part_area * num_rows / (p_base * w_z)) * 100
                        cost_z = (((p_base * w_z * material_thickness) * material_density) / 1_000_000) * material_price / num_rows
                        record_z = {'util': util_z, 'cost': cost_z, 'angle': angle, 'parts': z_parts, 'w': w_z, 'p': p_base}
                        zigzag_results.append({
                            '각도': f"{angle}°", '피치(mm)': round(p_base, 2), '소재폭(mm)': round(w_z, 2),
                            '소재이용율(%)': round(util_z, 2), '1개당 원가(원)': int(cost_z), '압연방향 적합': 'O' if valid else 'X'
                        })
                        if fallback_z is None or util_z > fallback_z['util']: fallback_z = record_z
                        if valid and (best_z is None or util_z > best_z['util']): best_z = record_z

            s_fallback_used = best_s is None
            if best_s is None: best_s = fallback_s
            i_fallback_used = best_i is None
            if best_i is None: best_i = fallback_i
            z_fallback_used = best_z is None
            if best_z is None: best_z = fallback_z

            st.session_state.update({
                'part_area': part_area, 'holes_count': len(holes),
                'holes_area': sum(h.area for h in holes) if holes else 0.0,
                'single': (single_results, best_s, s_fallback_used),
                'inter': (inter_results, best_i, i_fallback_used),
                'zigzag': (zigzag_results, best_z, z_fallback_used),
                'unit_msg': unit_msg
            })
            st.session_state.last_params = current_params

    s_res, best_s, s_fall = st.session_state['single']
    i_res, best_i, i_fall = st.session_state['inter']
    z_res, best_z, z_fall = st.session_state['zigzag']
    
    st.caption(st.session_state['unit_msg'])
    if st.session_state['holes_count'] > 0:
        st.info(f"🕳️ 내측 홀 **{st.session_state['holes_count']}개** 인식됨 (홀 면적 합계: {st.session_state['holes_area']:.1f} mm²) → 소재이용율·원가 계산에 순단면적이 반영되었습니다.")

    candidates = [('단일 배열', best_s)]
    if best_i: candidates.append((f'{num_rows}열 교차 배열', best_i))
    if best_z: candidates.append((f'{num_rows}열 지그재그 배열', best_z))
    best_method_name, best_overall = min(candidates, key=lambda kv: kv[1]['cost'])
    
    saving_cost = int(best_s['cost'] - best_overall['cost'])
    st.success(f"🏆 분석 완료! 가장 훌륭한 배열은 **[{best_method_name}]**이며, 단일 배열 대비 1개당 :blue[**{saving_cost:,}원**]을 절감합니다. (캐리어 폭 {carrier_width}mm 포함 기준)")

    st.header("📊 [1단계] 단위 배열 최적화 결과")
    if apply_rolling_constraint:
        st.markdown("💡 **안내:** 표 내부의 <span style='background-color:#ffe6e6; padding:2px 6px; border-radius:4px;'>붉은색 행</span>은 압연방향(그레인) 이격각 조건을 불만족하여 **크랙(터짐) 불량 위험이 높은 기각(사용 불가) 배열**을 의미합니다.", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    # 렌더링에 사용될 다열용 색상 팔레트
    color_palette_i = ['#004b87', '#007934'] * 5 
    color_palette_z = ['#004b87', '#d55e00'] * 5 

    render_case_column(col1, "[1] 단일 배열", s_res, best_s, s_fall, ['#004b87'], margin, carrier_width)
    
    if best_i: render_case_column(col2, f"[2] {num_rows}열 교차 배열", i_res, best_i, i_fall, color_palette_i[:num_rows], margin, carrier_width)
    else: 
        with col2:
            st.subheader(f"[2] {num_rows}열 교차 배열")
            st.warning("교차 다열 배열 불가 형상")
    
    if best_z: render_case_column(col3, f"[3] {num_rows}열 지그재그 배열", z_res, best_z, z_fall, color_palette_z[:num_rows], margin, carrier_width)
    else: 
        with col3:
            st.subheader(f"[3] {num_rows}열 지그재그 배열")
            st.warning("지그재그 다열 배열 불가 형상")

    st.divider()
    st.header("🎞️ [2단계] 스트립 Layout도 및 금형 코어 사이즈 도출")
    st.markdown(f"좌측에서 입력하신 **총 {total_stations} 피치**, **캐리어 폭 {carrier_width}mm**, **파일럿 홀 ⌀{pilot_dia}mm** 조건을 반영한 실제 금형 내부 작업 구간 설계 도면입니다.")
    
    st.subheader("◼️ [1] 단일 배열 Layout도")
    render_strip_section("단일 배열", best_s, total_stations, margin, carrier_width, pilot_dia, ['#004b87'])
    
    st.divider()
    st.subheader(f"◼️ [2] {num_rows}열 교차 배열 Layout도")
    if best_i: 
        render_strip_section(f"{num_rows}열 교차 배열", best_i, total_stations, margin, carrier_width, pilot_dia, color_palette_i[:num_rows])
    else:
        st.warning(f"이 부품은 {num_rows}열 교차 배열이 불가능합니다.")
        
    st.divider()
    st.subheader(f"◼️ [3] {num_rows}열 지그재그 배열 Layout도")
    if best_z: 
        render_strip_section(f"{num_rows}열 지그재그 배열", best_z, total_stations, margin, carrier_width, pilot_dia, color_palette_z[:num_rows])
    else:
        st.warning(f"이 부품은 {num_rows}열 지그재그 배열이 불가능합니다.")


    # ============================================================
    # [7] 수동 미세 조정 (Fine-Tuning) - 다열 맞춤형 개편
    # ============================================================
    st.divider()
    st.header("🛠️ [3단계] 수동 미세 조정 (Fine-Tuning)")
    st.markdown("자동으로 계산된 최적 배열을 바탕으로 실무 설계자의 감각에 맞춰 **숫자를 증감(+, -) 시키면 즉시 도면과 원가에 반영**됩니다.")

    if 'last_tune_target' not in st.session_state:
        st.session_state.last_tune_target = None

    valid_option_names = [k for k, v in candidates]

    if ('tune_target_select' not in st.session_state
            or st.session_state['tune_target_select'] not in valid_option_names):
        st.session_state['tune_target_select'] = best_method_name

    tune_target_name = st.selectbox("조정할 배열 방식 선택", options=valid_option_names, key='tune_target_select')
    target_best = next(v for k, v in candidates if k == tune_target_name)
    is_multi = len(target_best['parts']) > 1

    tkey = tune_target_name

    tune_reset_signature = (file_hash, tune_target_name)
    if st.session_state.last_tune_target != tune_reset_signature or recalculated:
        st.session_state[f"tune_angle_{tkey}"] = float(target_best['angle'])
        st.session_state[f"tune_pitch_{tkey}"] = float(target_best['p'])
        st.session_state[f"tune_width_{tkey}"] = float(target_best['w'])
        st.session_state[f"tune_y_all_{tkey}"] = 0.0
        st.session_state[f"tune_x_step_{tkey}"] = 0.0
        st.session_state[f"tune_y_step_{tkey}"] = 0.0
        st.session_state.last_tune_target = tune_reset_signature

    fc1, fc2 = st.columns(2)
    with fc1:
        st.subheader("⚙️ 파라미터 미세조정")
        tune_angle = st.number_input("전체 회전 각도 (°, 원본 기준)", step=1.0, key=f"tune_angle_{tkey}")
        tune_pitch = st.number_input("피치 (Pitch, mm)", step=0.1, min_value=0.1, key=f"tune_pitch_{tkey}")
        tune_width = st.number_input("소재 폭 (Width, mm)", step=0.5, min_value=0.1, key=f"tune_width_{tkey}")

    with fc2:
        st.subheader("↕️ 위치 오프셋 (Offset)")
        tune_y_all = st.number_input("전체 Y축 위치 이동 (mm)", step=0.5, key=f"tune_y_all_{tkey}")
        tune_x_step = st.number_input("행간 X축 간격 추가 조절 (mm)", step=0.5, disabled=not is_multi, key=f"tune_x_step_{tkey}")
        tune_y_step = st.number_input("행간 Y축 간격 추가 조절 (mm)", step=0.5, disabled=not is_multi, key=f"tune_y_step_{tkey}")

    # 다열 배열에 대응하는 미세 조정 형상 재계산
    center_geom = target_best['parts'][0] if len(target_best['parts']) == 1 else unary_union(target_best['parts'])
    delta_angle = tune_angle - target_best['angle']
    tuned_parts = []
    
    for idx, geom in enumerate(target_best['parts']):
        g = rotate(geom, delta_angle, origin=center_geom.centroid)
        g = translate(g, 0, tune_y_all)  # 1. 전체 그룹 Y축 이동
        g = translate(g, idx * tune_x_step, idx * tune_y_step) # 2. 행(Row)간 추가 간격(벌리기/좁히기)
        tuned_parts.append(g)

    all_geom_tuned = unary_union(tuned_parts)
    minx, miny, maxx, maxy = all_geom_tuned.bounds

    interference_tol = 0.01  
    is_clashing = False
    
    base_min_pitch = calculate_1d_pitch(tuned_parts[0], bridge)
    if tune_pitch < base_min_pitch - interference_tol:
        is_clashing = True
    else:
        # 다열 배열 간섭 검증
        for r in range(1, len(tuned_parts)):
            buf_prev = tuned_parts[r-1].buffer(bridge - interference_tol, resolution=4)
            if buf_prev.intersects(tuned_parts[r]): is_clashing = True
            
        for r in range(len(tuned_parts)):
            buf_r = tuned_parts[r].buffer(bridge - interference_tol, resolution=4)
            for step in [-1, 1]:
                if buf_r.intersects(translate(tuned_parts[r], xoff=step*tune_pitch, yoff=0)):
                    is_clashing = True
        
        for r in range(1, len(tuned_parts)):
            buf_prev = tuned_parts[r-1].buffer(bridge - interference_tol, resolution=4)
            for step in [-1, 1]:
                if buf_prev.intersects(translate(tuned_parts[r], xoff=step*tune_pitch, yoff=0)):
                    is_clashing = True

    if is_clashing:
        st.error(f"🚫 **간섭 경고:** 현재 설정된 피치({tune_pitch:.2f}mm) 또는 오프셋 위치에서는 인접한 부품끼리 겹치거나 최소 브릿지 간격({bridge}mm)을 침범합니다. 값을 넉넉하게 조정하세요.")
        
    min_required_width = (maxy - miny) + margin * 2 + carrier_width * 2
    if tune_width < min_required_width - interference_tol:
        st.error(f"🚫 **폭 부족 경고:** 현재 소재 폭({tune_width:.2f}mm)이 이 형상을 담기 위한 "
                 f"최소 폭({min_required_width:.2f}mm)보다 작습니다. 부품이 마진/캐리어 영역을 벗어날 수 있습니다.")

    # 원가 재계산
    tune_util = (st.session_state['part_area'] * len(tuned_parts)) / (tune_pitch * tune_width) * 100
    tune_cost = (((tune_pitch * tune_width * material_thickness) * material_density) / 1_000_000) * material_price / len(tuned_parts)

    st.success(f"**미세조정 결과** ➔ 변경된 소재이용율: :blue[**{tune_util:.2f}%**]  |  변경된 1개당 단가: :blue[**{int(tune_cost):,}원**]")

    # 화면 렌더링용 기하학 계산
    part_length_tuned = maxx - minx
    total_length_tuned = (tune_pitch * (total_stations - 1)) + part_length_tuned + (tune_pitch * 0.4)
    x_shift = -minx + (tune_pitch * 0.2)
    extra_width = max(0.0, tune_width - min_required_width)
    y_shift = -miny + margin + carrier_width + extra_width / 2

    fig_tune, ax_tune = plt.subplots(figsize=(max(8, total_stations * 2), 4))
    
    ax_tune.plot([0, total_length_tuned, total_length_tuned, 0, 0], 
                 [0, 0, tune_width, tune_width, 0], color='red', linestyle='-', linewidth=2.5,
                 label=f'조정 후 코어 사이즈\n(가로: {total_length_tuned:.1f} x 세로: {tune_width:.1f} | 피치: {tune_pitch:.2f})')
    
    if carrier_width > 0:
        ax_tune.add_patch(Rectangle((0, 0), total_length_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))
        ax_tune.add_patch(Rectangle((0, tune_width - carrier_width), total_length_tuned, carrier_width, facecolor='#999999', alpha=0.25, edgecolor='none', zorder=1))

    for i in range(total_stations):
        for idx, geom in enumerate(tuned_parts):
            if "교차" in tune_target_name:
                color = color_palette_i[idx % len(color_palette_i)]
            elif "지그재그" in tune_target_name:
                color = color_palette_z[idx % len(color_palette_z)]
            else:
                color = '#004b87'
                
            shifted = translate(geom, xoff=x_shift + (i * tune_pitch), yoff=y_shift)
            plot_polygon(ax_tune, shifted, color, lw=1.5, alpha=0.7)

        if carrier_width > 0 and 0 < pilot_dia < carrier_width:
            ax_tune.add_patch(Circle((tune_pitch * (i + 0.5), carrier_width / 2), pilot_dia / 2, facecolor='white', edgecolor='black', linewidth=1.2, zorder=5))
        if i < total_stations - 1:
            ax_tune.plot([tune_pitch * (i + 1), tune_pitch * (i + 1)], [0, tune_width], color='black', linestyle=':', alpha=0.4, zorder=1)

    ax_tune.axis('equal'); ax_tune.set_xticks([]); ax_tune.set_yticks([])
    ax_tune.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    st.pyplot(fig_tune)

    # ============================================================
    # [8] 데이터 추출 및 내보내기 (DXF / Excel / PDF)
    # ============================================================
    st.divider()
    st.header("💾 [4단계] 데이터 추출 및 내보내기")
    st.markdown("최종 미세조정(Fine-Tuning)이 완료된 결과를 바탕으로 **CAD 도면(DXF)**, **견적용 데이터(Excel)**, 그리고 **보고용 문서(PDF)**를 다운로드할 수 있습니다.")
    
    dxf_bytes = generate_dxf_bytes(
        tuned_parts=tuned_parts, tune_pitch=tune_pitch, tune_width=tune_width,
        total_stations=total_stations, margin=margin, carrier_width=carrier_width,
        pilot_dia=pilot_dia, x_shift=x_shift, y_shift=y_shift
    )
    
    excel_bytes = generate_excel_report(
        tune_target_name, tune_pitch, tune_width, tune_util, tune_cost, 
        total_stations, mat_type, material_thickness, material_price, material_density,
        s_res, i_res, z_res
    )
    
    pdf_bytes = generate_pdf_report(
        fig_tune, tune_target_name, tune_pitch, tune_width, tune_util, tune_cost, 
        total_stations, mat_type, material_thickness
    )
    
    col_dxf, col_xls, col_pdf = st.columns(3)
    
    with col_dxf:
        st.subheader("📐 CAD 도면")
        st.download_button(label="📥 DXF 다운로드", data=dxf_bytes, file_name="optimized_strip_layout.dxf", mime="application/dxf", type="primary", use_container_width=True)
        
    with col_xls:
        st.subheader("📊 견적용 데이터")
        st.download_button(label="📥 Excel 다운로드", data=excel_bytes, file_name="layout_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
        
    with col_pdf:
        st.subheader("📄 보고용 문서")
        st.download_button(label="📥 PDF 다운로드", data=pdf_bytes, file_name="layout_report.pdf", mime="application/pdf", type="primary", use_container_width=True)
