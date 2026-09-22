import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import './style.css'

const P = [
  { id:'A', x:2, y:3 }, { id:'B', x:5, y:18 }, { id:'C', x:8, y:7 },
  { id:'D', x:9, y:8 }, { id:'E', x:13, y:17 }, { id:'F', x:20, y:3 }
]
const splitX = 8.5
const d0 = Math.sqrt(52)
const pseudo = [
  'closestPair(Px, Py):',
  '  sort Px theo x, Py theo y (một lần)',
  '  chia Px; tách Py tương ứng sang L / R',
  '  dL = closestPair(L); dR = closestPair(R)',
  '  d = min(dL, dR)',
  '  strip = lọc từ Py với |x - midX| < d',
  '  mỗi điểm so tối đa 7 điểm kế tiếp',
  '  return cặp tốt nhất',
  'T(n) = 2T(n/2) + O(n)',
  '=> O(n log n)'
]
const steps = [
  ['Bài toán','Tìm 2 điểm gần nhau nhất','problem',['Cho n điểm trên mặt phẳng.','Tìm cặp có khoảng cách Euclid nhỏ nhất.','Vét cạn xét n(n−1)/2 cặp, tức O(n²).'],'dist(P,Q) = √((x₁−x₂)² + (y₁−y₂)²)','Nếu thử mọi cặp thì số phép so sánh tăng rất nhanh. Chia để trị giúp ta bỏ qua phần lớn cặp không cần thiết.',0],
  ['Sắp xếp','Chuẩn bị để chia đôi','sorted',['Tạo Px sắp theo x và Py sắp theo y.','Theo x: A → B → C → D → E → F.','Sort chỉ làm một lần: O(n log n).'],'Px = sortByX(P), Py = sortByY(P)','Ta sort trước để biết chính xác nửa trái, nửa phải và vẫn giữ được thứ tự y cho bước gộp.',1],
  ['Divide','Chia thành 2 nửa','divide',['Cắt tại đường giữa x = 8.5.','Trái: A, B, C. Phải: D, E, F.','Mỗi bài toán con có kích thước gần n/2.'],'T(n) → T(n/2) + T(n/2)','Đây là Divide: biến một bài toán lớn thành hai bài toán cùng dạng nhưng nhỏ hơn.',2],
  ['Conquer trái','Giải bài toán con L','left',['Trong A, B, C, cặp gần nhất là A–C.','dL = √52 ≈ 7.21.','Với ít điểm ta kiểm tra trực tiếp.'],'dL = dist(A,C) = √52 ≈ 7.21','Ta giải nửa trái và chỉ giữ kết quả tốt nhất của nửa này.',3],
  ['Conquer phải','Giải bài toán con R','right',['Trong D, E, F, cặp gần nhất là D–E.','dR = √97 ≈ 9.85.','Nửa phải cũng trả về một kết quả tốt nhất.'],'dR = dist(D,E) = √97 ≈ 9.85','Tương tự, nửa phải trả kết quả lên cho bước gộp.',3],
  ['Lấy d hiện tại','d = min(dL, dR)','min',['So sánh hai kết quả con.','d = min(7.21, 9.85) = 7.21.','Chưa thể kết luận A–C là đáp án cuối.'],'d = min(dL, dR) = 7.21','Cặp gần nhất thật sự có thể gồm một điểm bên trái và một điểm bên phải.',4],
  ['Combine: strip','Chỉ xét gần đường chia','strip',['Giữ điểm có |x − midX| < d.','Điểm quá xa đường chia không thể tạo cặp tốt hơn d.','Lọc từ Py nên strip đã có thứ tự y.'],'strip = { P | |P.x − 8.5| < 7.21 }','Strip là mẹo quyết định: ta không so mọi điểm trái với mọi điểm phải.',5],
  ['Quét strip','Phát hiện cặp qua biên','scan',['Xét các điểm trong strip theo y.','Mỗi điểm chỉ cần so tối đa 7 điểm kế tiếp.','C–D có khoảng cách √2 ≈ 1.41, nhỏ hơn 7.21.'],'dist(C,D) = √((9−8)² + (8−7)²) = √2','C và D nằm khác nửa nhưng lại là cặp gần nhau nhất toàn bộ tập điểm.',6],
  ['Kết quả','Closest Pair = C–D','result',['C(8,7) và D(9,8) là cặp gần nhất.','Khoảng cách cuối = √2 ≈ 1.414.','Kết quả được cập nhật ở bước Combine.'],'answer = (C,D), dmin = √2 ≈ 1.414','Combine là phần không thể bỏ qua vì cặp tốt nhất có thể nằm qua hai phía đường chia.',7],
  ['Phân tích T(n)','Từ O(n²) xuống O(n log n)','complexity',['Hai bài toán con: 2T(n/2).','Tách Py, lọc và quét strip: O(n).','Master Theorem cho O(n log n).'],'T(n) = 2T(n/2) + O(n) ⇒ O(n log n)','Lưu ý: phải giữ Py đã sort theo y; nếu sort strip lại ở mỗi tầng có thể thành O(n log² n).',8],
  ['Đánh giá & ứng dụng','Khi nào nên dùng?','use',['Dùng khi dữ liệu điểm lớn và cần đáp án chính xác.','Ứng dụng: GIS, robot, game, clustering, hình học tính toán.','Nhược điểm: cài đặt khó hơn vét cạn, nhất là bước strip.'],'Thời gian: O(n log n) · Bộ nhớ: O(n)','Thuật toán này thể hiện đủ Divide, Conquer, Combine và cho thấy rõ lợi ích giảm độ phức tạp.',9]
]

const W=760, H=430, PAD=54, XMAX=22, YMAX=21
const sx=x=>PAD+(x/XMAX)*(W-PAD*2)
const sy=y=>H-PAD-(y/YMAX)*(H-PAD*2)
const point=id=>P.find(p=>p.id===id)

function Pair({ ids, primary=false, label='' }) {
  const a=point(ids[0]), b=point(ids[1]), mx=(sx(a.x)+sx(b.x))/2, my=(sy(a.y)+sy(b.y))/2
  return <g className={primary?'pair primary':'pair'}>
    <line x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)} />
    <rect x={mx-30} y={my-17} width="60" height="24" rx="6" />
    <text x={mx} y={my} textAnchor="middle">{label}</text>
  </g>
}

function Plot({ stage }) {
  const divide=['divide','left','right','min','strip','scan','result','complexity'].includes(stage)
  const showL=['left','min','strip','scan'].includes(stage)
  const showR=['right','min','strip','scan'].includes(stage)
  const strip=['strip','scan','result'].includes(stage)
  const final=['scan','result'].includes(stage)
  const left=Math.max(0,splitX-d0), right=Math.min(XMAX,splitX+d0)
  return <svg className="plot" viewBox={'0 0 '+W+' '+H} role="img" aria-label="Mặt phẳng tọa độ minh họa Closest Pair">
    <defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" className="gridline"/></pattern></defs>
    <rect className="plotbg" width={W} height={H} rx="13"/>
    <rect x={PAD} y={PAD} width={W-PAD*2} height={H-PAD*2} fill="url(#grid)"/>
    {strip && <rect className="strip" x={sx(left)} y={PAD} width={sx(right)-sx(left)} height={H-PAD*2}/>}
    {divide && <g className="divider"><line x1={sx(splitX)} y1={PAD} x2={sx(splitX)} y2={H-PAD}/><text x={sx(splitX)+10} y={PAD+20}>midX = 8.5</text></g>}
    <line className="axis" x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD}/><line className="axis" x1={PAD} y1={PAD} x2={PAD} y2={H-PAD}/>
    {showL && <Pair ids={['A','C']} label="7.21"/>}{showR && <Pair ids={['D','E']} label="9.85"/>}{final && <Pair ids={['C','D']} primary label="1.41"/>}
    {P.map(p=>{
      const faded=(stage==='left'&&p.x>splitX)||(stage==='right'&&p.x<splitX), hot=final&&['C','D'].includes(p.id)
      return <g key={p.id} className={'pt '+(faded?'fade ':'')+(hot?'hot':'')} transform={'translate('+sx(p.x)+','+sy(p.y)+')'}>
        <circle r={hot?10:8}/><text className="pid" x="13" y="-10">{p.id}</text><text className="coord" x="13" y="8">({p.x},{p.y})</text>
      </g>
    })}
    {stage==='sorted' && <text className="order" x="92" y="34">A → B → C → D → E → F</text>}
  </svg>
}

function App() {
  const [i,setI]=useState(0), [auto,setAuto]=useState(false), [cue,setCue]=useState(true)
  const s=steps[i], last=steps.length-1
  const move=d=>{setAuto(false);setI(v=>Math.min(last,Math.max(0,v+d)))}
  const fullscreen=()=>document.fullscreenElement?document.exitFullscreen?.():document.documentElement.requestFullscreen?.()

  useEffect(()=>{
    if(!auto) return
    if(i===last){setAuto(false);return}
    const t=setTimeout(()=>setI(v=>Math.min(last,v+1)),5000)
    return()=>clearTimeout(t)
  },[auto,i,last])

  useEffect(()=>{
    const key=e=>{
      if(e.key==='ArrowRight')move(1)
      else if(e.key==='ArrowLeft')move(-1)
      else if(e.key===' '){e.preventDefault();setAuto(v=>!v)}
      else if(e.key.toLowerCase()==='f')fullscreen()
      else if(e.key.toLowerCase()==='n')setCue(v=>!v)
      else if(e.key.toLowerCase()==='r'){setAuto(false);setI(0)}
    }
    addEventListener('keydown',key); return()=>removeEventListener('keydown',key)
  },[])

  return <main className="shell">
    <header><div><p className="eyebrow">Chia để trị · Demo thuật toán</p><h1>Closest Pair of Points</h1></div><div className="head-actions"><button onClick={()=>setCue(v=>!v)}>{cue?'Ẩn lời thoại':'Hiện lời thoại'}</button><button onClick={fullscreen}>Toàn màn hình</button></div></header>
    <div className="progress"><i style={{width:((i+1)/steps.length*100)+'%'}}/></div>
    <section className="layout">
      <div className="visual">
        <div className="visual-title"><div><span>Bước {i+1}/{steps.length}</span><h2>{s[0]}</h2></div><p>{s[1]}</p></div>
        <div className="plotbox"><Plot stage={s[2]}/></div>
        <nav className="rail" aria-label="Các bước">{steps.map((x,n)=><button key={x[0]} className={n===i?'active':''} onClick={()=>{setAuto(false);setI(n)}} aria-label={'Bước '+(n+1)+': '+x[0]}>{n+1}</button>)}</nav>
      </div>
      <aside>
        <section><h3>Điều đang xảy ra</h3><ul>{s[3].map(x=><li key={x}>{x}</li>)}</ul></section>
        <section className="formula"><small>Công thức</small><strong>{s[4]}</strong></section>
        <section><div className="code-head"><h3>Mã giả</h3><small>Dòng đang chạy</small></div><pre>{pseudo.map((x,n)=><code key={x} className={n===s[6]?'on':''}><b>{String(n+1).padStart(2,'0')}</b>{x}</code>)}</pre></section>
        {cue && <section className="cue"><small>Bạn nói</small><p>{s[5]}</p></section>}
      </aside>
    </section>
    <footer><button disabled={i===0} onClick={()=>move(-1)}>← Bước trước</button><span>← → chuyển bước · Space tự chạy · F toàn màn hình · N lời thoại</span><button className="auto" onClick={()=>setAuto(v=>!v)}>{auto?'Dừng':'Tự chạy'}</button><button className="next" disabled={i===last} onClick={()=>move(1)}>Bước tiếp →</button></footer>
  </main>
}

ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>)
