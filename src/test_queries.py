"""
test_queries.py
---------------
Curated orthopaedic clinical queries and expert ground-truth answers
used for final evaluation of the RAG pipeline.

Domain: Total / Partial / Unicompartmental Knee Arthroplasty
Source: IITG-AIIMS Knee Arthroplasty RAG project
"""

TEST_QUERIES = [
    "During total knee replacement, the extension gap is tight and the knee cannot be fully extended. At more than 90 degrees of flexion, the tibial tray lifts off, indicating a tight flexion gap as well. Both the flexion and extension gaps are equally tight. What should be done to solve this problem?",
    "A 40-year-old man presents with groin pain. X-ray shows avascular necrosis of the femoral head with subchondral lucency but no collapse. He is at risk for progression to collapse. How does bisphosphonate therapy help prevent this?",
    "Enumerate the most important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and fixed flexion deformity.",
    "Enumerate the most important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and Valgus deformity.",
    "What percentage of patients with complete peroneal nerve palsy after total hip arthroplasty will never recover full strength?",
    "A 68-year-old patient presents 8 months after total knee arthroplasty with complaints of giving way while descending stairs and recurrent swelling. What is your differential diagnosis?",
    "What investigations are required to assess patellar maltracking after TKA?",
    "A 68-year-old male presents 8 months after undergoing Total knee arthroplasty. He complains of a painful catching sensation in the knee while rising from a chair. He describes a distinct clunk when extending his knee from a flexed position. What is the diagnosis. How to treat this condition?",
    "Before doing a total knee replacement what pre operative investigations should I do especially in a patient with diabetes mellitus",
    "What is the difference between measured resection and gap balancing in total knee replacement",
    "what are the conservative treatment measures in a patient with ostearthritis of knee",
    "What is inlay vs onlay patellar implant",
    "What is Bristol patellar wear score",
    "What is fellar patellar score?",
    "I am having a knee with extra-articular deformity of 40 degrees in the tibia. I want to do a total knee replacement. What principles should I follow to proceed with TKR",
]

GROUND_TRUTHS = [
    """In a total knee replacement, inability to fully extend the knee indicates a tight extension gap, and tibial tray lift-off beyond 90 degrees of flexion indicates a tight flexion gap; therefore, both flexion and extension gaps are tight. When both gaps are equally tight, the appropriate intervention is additional proximal tibial resection because the tibial cut affects both the flexion and extension gaps equally. By recutting the proximal tibia slightly, both gaps are increased uniformly, restoring full extension, eliminating tibial tray lift-off in flexion, and achieving a balanced knee throughout the range of motion.""",

    """A 40-year-old man with groin pain and radiographic evidence of femoral head avascular necrosis showing subchondral lucency without collapse represents early (pre-collapse) disease, in which the necrotic trabecular bone becomes structurally weak and prone to subchondral fracture and eventual collapse. The progression to collapse occurs largely due to increased osteoclastic resorption during the revascularization phase, which further weakens the already compromised bone. Bisphosphonate therapy has been shown to decrease the risk of femoral head collapse in this stage by inhibiting osteoclast-mediated bone resorption, thereby preserving trabecular architecture and maintaining subchondral bone strength. By slowing excessive bone turnover and structural deterioration, bisphosphonates help delay or prevent progression to collapse and may postpone the need for surgical intervention.""",

    """The five most important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and fixed flexion deformity depend on the severity of the deformity. For mild deformity, the key steps are excision of medial and posterior osteophytes to remove bony blocks to extension, along with posteromedial soft tissue release to correct tightness contributing to the flexion contracture. For moderate deformity, additional steps include posterior capsular release to address persistent extension tightness, decreasing the tibial slope to improve extension balance, releasing up to the tendinous origin of the gastrocnemius when required, and performing pie-crusting of the medial collateral ligament (MCL) to correct residual medial tightness. For severe deformity, more extensive correction is needed, including an extra distal femoral cut to increase the extension gap, medial epicondylar osteotomy to facilitate adequate soft tissue balancing, and, when instability persists despite releases, the use of constrained implants to achieve a stable and well-balanced knee.""",

    """In total knee arthroplasty for a valgus knee deformity, a lateral parapatellar approach can be used to facilitate direct access to the contracted lateral structures. The tibial resection is performed in the standard manner, without alteration. For the femur, a 3 degree valgus distal femoral cut is taken to help restore appropriate alignment. Soft tissue balancing is critical and follows the Ranawat inside-out release technique, beginning with removal of lateral osteophytes, followed by sequential release of the PCL (if required), posterolateral capsule, iliotibial band, further posterolateral capsular release, and popliteus release as needed. During flexion gap balancing, the epicondylar axis is used as the primary reference for femoral component rotation, along with the cut surface of the tibia, while the posterior condylar reference is usually not relied upon because it is unreliable in valgus knees due to lateral condylar hypoplasia. An alternative technique for balancing is lateral epicondylar osteotomy when additional correction is required.""",

    """Approximately 40 to 50 percent of patients with complete peroneal nerve palsy after total hip arthroplasty will never recover full strength.""",

    """A 68-year-old patient presenting 8 months after total knee arthroplasty with complaints of giving way while descending stairs and recurrent swelling raises concern for several possibilities. The differential diagnosis includes flexion instability (especially mid-flexion or late flexion instability, which commonly presents with giving way on stairs), component malposition or malrotation leading to imbalance, polyethylene wear or early mechanical loosening, extensor mechanism insufficiency, patellofemoral instability or maltracking, periprosthetic joint infection (chronic low-grade infection presenting with recurrent effusion), and aseptic loosening of components.""",

    """Skyline (Merchant) view, CT scan (to assess femoral and tibial component rotation), Long-leg alignment films.""",

    """Diagnosis: Patellar clunk syndrome. Treatment: Initial management may include observation if symptoms are mild. Definitive treatment is arthroscopic or open debridement of the fibrous nodule at the superior pole of the patella/posterior quadriceps tendon that catches in the intercondylar box of the femoral component. This typically relieves the catching sensation and clunk.""",

    """Before total knee replacement in a patient with Diabetes Mellitus, pre-operative evaluation includes routine tests such as CBC, renal and liver function tests, serum electrolytes, coagulation profile, blood grouping, urine analysis and culture, ECG, and chest X-ray, along with joint-specific X-rays; additionally, strict assessment of glycemic control with fasting and postprandial blood sugar and HbA1c (ideally <7-8%) is essential, and screening for diabetic complications should be done including renal function (creatinine, eGFR, urine albumin), cardiovascular status (ECG +/- echocardiography or stress test), and infection foci (urinary, dental, skin/foot), along with inflammatory markers like ESR and CRP, since uncontrolled diabetes and occult infections significantly increase the risk of postoperative complications such as wound infection and poor healing.""",

    """In total knee replacement, measured resection and gap balancing are two techniques used to achieve proper alignment and soft-tissue balance: in measured resection, bone cuts are made first based on anatomical landmarks (such as femoral condyles, transepicondylar axis, and tibial alignment), and the soft tissues are then adjusted secondarily to fit the prosthesis, making it more anatomy-driven but potentially less accurate in cases with deformity; in contrast, gap balancing focuses first on achieving equal and rectangular flexion and extension gaps by sequential soft-tissue releases, and bone cuts are then tailored to maintain these balanced gaps, making it more ligament-driven and often preferred in cases with severe deformity or instability, as it provides better soft-tissue balance but may alter native anatomy.""",

    """Conservative management of Osteoarthritis of the knee focuses on pain relief and functional improvement and includes lifestyle measures such as weight reduction and activity modification (avoiding squatting, prolonged standing, and high-impact activities), physiotherapy with quadriceps strengthening and range-of-motion exercises, use of assistive devices like a cane or walker, and knee braces; pharmacological options include paracetamol, NSAIDs (topical or oral), and intra-articular injections like corticosteroids or hyaluronic acid, while supportive therapies such as heat therapy, hydrotherapy (water-based exercises), sauna baths, and other forms of thermotherapy help reduce pain and stiffness, along with footwear modification and patient education as integral parts of non-surgical care.""",

    """Inlay patellar components are inserted into a reamed cavity within the patella, preserving surrounding bone for stability, while onlay components are placed on the cut retropatellar surface. Onlay implants are larger and more tolerant of cutting errors; inlay designs may theoretically reduce patellar tilt and shift but carry a higher fracture risk if excessive cancellous bone is removed. A minimum residual patellar thickness of 15 mm is recommended for inlay designs. Clinical evidence of superior outcomes with inlay implants remains limited.""",

    """The Bristol Patella Wear Score divides the retropatellar surface into four zones graded 0 (normal cartilage), 1 (softened cartilage), 2 (fibrillated/fissured cartilage), and 3 (exposed bone), giving a maximum total score of 12.""",

    """The Feller Patellar Score evaluates four domains: anterior knee pain (0-15), quadriceps strength (1-5), ability to rise from a chair (0-5), and stair climbing (2-5), with a maximum total score of 30.""",

    """In a patient with severe extra-articular tibial deformity (~40 degrees) undergoing total knee replacement, the key principle is to determine whether the deformity can be corrected intra-articularly or requires an extra-articular osteotomy; generally, deformities >20-30 degrees cannot be safely corrected by bone cuts alone without compromising ligament balance, so a staged or simultaneous corrective osteotomy with TKR is usually indicated; careful pre-operative planning with long-leg alignment views is essential to restore the mechanical axis, and during surgery, preservation of collateral ligament attachments, achieving balanced flexion-extension gaps, and proper component positioning are critical; intramedullary guides may be unreliable in deformed tibia, so extramedullary guides or navigation/robotics are preferred, and the use of stems or constrained prostheses may be required for stability, while ensuring gradual deformity correction to avoid neurovascular complications.""",
]

assert len(TEST_QUERIES) == len(GROUND_TRUTHS), (
    f"Mismatch: {len(TEST_QUERIES)} queries vs {len(GROUND_TRUTHS)} ground truths"
)

print(f"Test queries     : {len(TEST_QUERIES)}")
print(f"Ground truths    : {len(GROUND_TRUTHS)}")
