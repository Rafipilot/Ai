import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import time

# Sample data related to rockets
rocket_data = [
   "Interplanetary travel requires advanced rocket propulsion systems.",
    "Rocket scientists analyze data from telemetry to improve launch success rates.",
    "SpaceX's Starship aims to revolutionize space travel with its fully reusable design.",
    "The concept of space debris tracking is crucial for preventing collisions with active satellites and rockets.",
    "NASA's Mars rovers, including Opportunity and Spirit, paved the way for scientific discoveries on the Red Planet.",
    "Rocket launches from the Vandenberg Space Force Base contribute to a range of polar orbit missions.",
    "The concept of nuclear-powered rockets is explored for potential deep-space missions.",
    "SpaceX's Crew Dragon is a crewed spacecraft designed for missions to the International Space Station.",
    "The Bezos-backed Blue Origin focuses on developing reusable rocket technologies.",
    "Satellite constellations, like Starlink by SpaceX, aim to provide global internet coverage using rocket launches.",
    "The concept of a space race in the 21st century involves private companies competing for space exploration milestones.",
    "Ion thrusters, common in satellite propulsion, provide efficient and low-thrust maneuvers in space.",
    "The study of space geology involves analyzing rocks and minerals found on other celestial bodies.",
    "Rocket launch weather forecasts are crucial for ensuring safe and successful missions.",
    "The Artemis program plans to land the first woman and the next man on the Moon using the Space Launch System.",
    "Microgravity research on rockets contributes to understanding its effects on biological organisms.",
    "The dream of interstellar travel involves developing rockets capable of reaching other star systems.",
    "Rocket landing platforms at sea, like SpaceX's drone ships, enable reusable rocket stages to return safely.",
    "International partnerships, such as the ISS collaboration, showcase diplomatic efforts in space exploration.",
    "CubeSats are small, cube-shaped satellites often launched as secondary payloads on rockets.",
    "Rocket fairing recovery aims to reduce space debris by retrieving the protective shells after launch.",
    "The concept of space farming explores growing crops on spacecraft and other celestial bodies.",
    "Hybrid rocket engines, combining liquid and solid propulsion, offer advantages in safety and simplicity.",
    "NASA's Perseverance rover, launched by the Mars 2020 mission, searches for signs of past microbial life.",
    "SpaceX's Falcon 9 rocket is a workhorse for launching satellites, cargo, and crewed missions.",
    "Astrophysicists use data from rocket-borne instruments to study cosmic phenomena beyond Earth's atmosphere.",
    "Rocket ignition systems play a critical role in the reliable and timely start of rocket engines.",
    "The concept of a space bus involves a reusable spacecraft that shuttles between Earth and orbiting space stations.",
    "Rocket launch countdowns involve a sequence of carefully timed events leading up to liftoff.",
    "The study of space habitats explores sustainable living solutions for long-duration space missions.",
    "In-space manufacturing using 3D printers transported by rockets opens new possibilities for construction in orbit.",
    "Rocket engines using green propellants aim to minimize environmental impact during space travel.",
    "Space agencies use sounding rockets for scientific experiments in the upper atmosphere and ionosphere.",
    "The concept of a space telescope constellation involves multiple telescopes working together for enhanced observations.",
    "Rocket launches contribute to international efforts to monitor climate change and environmental conditions.",
    "The Space Launch System's powerful boosters provide the thrust needed for deep-space missions beyond Earth's orbit.",
    "Space tourism companies offer suborbital flights, allowing civilians to experience the sensation of weightlessness.",
    "Rocket-based exploration of ocean worlds, like Europa and Titan, aims to uncover secrets beneath their icy surfaces.",
    "Asteroid deflection missions using rockets are proposed to mitigate potential threats to Earth from near-Earth objects.",
    "Rocket-powered landers are essential for safely delivering payloads to the surfaces of other planets and moons.",
    "SpaceX's Super Heavy rocket, under development, is intended to carry heavy payloads for interplanetary missions.",
    "The concept of a space zoo involves transporting animals into orbit for scientific studies on the effects of microgravity.",
    "Rocket engine failures are meticulously studied to improve safety measures and prevent recurrence.",
    "SpaceX's Crew Dragon Demo-2 marked the first crewed orbital launch by a private company.",
    "The study of celestial mechanics helps optimize rocket trajectories for efficient space travel.",
    "Rocket payload separation mechanisms ensure that satellites and instruments are deployed at the correct orbital positions.",
    "The concept of a space train envisions interconnected spacecraft shuttling between Earth, the Moon, and Mars.",
    "Rocket launch pollution studies aim to understand and minimize the environmental impact of space missions.",
    "The study of space sociology explores the dynamics of human societies living in confined spacecraft environments.",
    "The concept of a space marathon involves running in simulated low-gravity environments aboard spacecraft.",
    "Rocket stages left in orbit may contribute to space debris, and efforts are made to deorbit or control their trajectories.",
    "SpaceX's Starlink mega-constellation, consisting of thousands of satellites, aims to provide global broadband internet coverage.",
    "The concept of space insurance involves assessing risks and providing coverage for satellite launches and space missions.",
    "Rocket engine gimbal systems allow for precise control of the rocket's orientation during flight.",
    "Space agencies collaborate on international space law to regulate activities and prevent conflicts in outer space.",
    "Rocket-based studies of cosmic rays contribute to understanding high-energy particles from distant astrophysical sources.",
    "The concept of a space newspaper involves transmitting news and updates to space stations and interplanetary outposts.",
    "Rocket-powered space probes, like Voyager 1 and 2, continue to send data from the outer reaches of the solar system.",
    "SpaceX's Mars colonization plans aim to establish a self-sustaining human settlement on the Red Planet.",
    "Rocket launch tourism offers enthusiasts the opportunity to witness live rocket launches from spaceports around the world.",
    "The concept of a space high-speed rail involves magnetically propelled spacecraft for rapid transport between celestial bodies.",
    "Rocket engine health monitoring systems provide real-time data to ensure safe and reliable operation.",
    "Space agencies study the effects of microgravity on human health to prepare for long-duration missions beyond Earth.",
    "Rocket-powered deep-space missions explore the outer reaches of the solar system and beyond.",
    "The concept of a space time capsule involves sending messages and artifacts to distant star systems for potential extraterrestrial discovery.",
    "Rocket-powered spacewalks enable astronauts to repair and maintain spacecraft in the harsh conditions of outer space.",
    "SpaceX's plans for a crewed mission around the Moon involve the Starship spacecraft and the Space Launch System.",
    "Rocket-based exploration of the Sun involves launching solar observatories to study solar activity and phenomena.",
    "The concept of a space theme park involves recreating the experience of space travel through simulated rocket rides and exhibits.",
    "Rocket-powered space drones are proposed for exploring the atmospheres of other planets and moons.",
    "Space agencies collaborate on space traffic management to avoid collisions and congestion in Earth's orbit.",
    "Rocket engine acoustic studies aim to minimize the noise produced during launches for environmental and safety reasons.",
    "Rockets are powerful machines that can travel into space.",
    "NASA has launched numerous rockets to explore the cosmos.",
    "Rocket propulsion relies on the principle of Newton's third law.",
    "Elon Musk's SpaceX company is known for developing reusable rockets.",
    "The Apollo program achieved historic moon landings using powerful rockets.",
    "Rocket science involves complex engineering and physics principles.",
    "Saturn V was a powerful rocket used in the Apollo missions.",
    "The Falcon Heavy, developed by SpaceX, is one of the most powerful operational rockets.",
    "Rocket engines use thrust to overcome Earth's gravitational pull during launch.",
    "The Hubble Space Telescope was carried into orbit by the Space Shuttle Discovery's rocket.",
    "Rocket technology has evolved significantly since the early days of space exploration.",
    "The International Space Station is resupplied by cargo rockets at regular intervals.",
    "Rocket launches are meticulously planned events with precise timing and coordination.",
    "The concept of multistage rockets allows for more efficient use of fuel during space travel.",
    "Rocket testing is a crucial step in ensuring the reliability and safety of space missions.",
    "Commercial companies, along with government agencies, are actively involved in rocket development.",
    "Advancements in materials science contribute to the design of lighter and more efficient rocket components.",
    "The Vega rocket, developed by the European Space Agency, is designed for small satellite launches.",
    "The James Webb Space Telescope was transported to its destination using an Ariane 5 rocket.",
    "Rocket payloads include satellites, scientific instruments, and sometimes even human explorers.",
    "SpaceX's Starship is a next-generation fully reusable spacecraft intended for interplanetary travel.",
    "Rocket launches can be viewed by the public at designated sites, providing a unique and awe-inspiring experience.",
    "Space agencies collaborate globally to share knowledge and resources for peaceful space exploration.",
    "The Delta IV Heavy is another example of a powerful rocket used for various space missions.",
    "Rocket failures are meticulously analyzed to improve safety and reliability in future launches.",
    "The concept of orbital mechanics is fundamental to understanding how rockets navigate in space.",
    "Reusable rocket technology has the potential to significantly reduce the cost of space exploration.",
    "The SLS (Space Launch System) is NASA's next-generation rocket designed for deep-space missions.",
    "In-space propulsion systems are used to maneuver satellites and spacecraft after they reach orbit.",
    "The development of ion propulsion systems represents a leap forward in fuel efficiency for long-duration space missions.",
    "The concept of rocketry dates back to ancient China, where gunpowder-filled tubes were used for propulsion.",
    "The X-15 rocket plane set numerous speed and altitude records during the early days of spaceflight research.",
    "Space agencies continuously explore innovative propulsion technologies for future missions.",
    "Rocket fuel is carefully selected based on the specific requirements and conditions of each mission.",
    "The concept of space elevators, though theoretical, presents an intriguing alternative to traditional rocket launches.",
    "Mars rovers, such as Curiosity and Perseverance, were transported to the Red Planet by specialized landing rockets.",
    "The study of astrodynamics is essential for predicting and optimizing the trajectories of rockets in space.",
    "Rocketry has played a pivotal role in advancing our understanding of the universe and our place in it.",
    "The concept of space tourism envisions rockets carrying civilians on suborbital or orbital journeys.",
    "The New Shepard rocket, developed by Blue Origin, is designed for suborbital space tourism.",
    "Rocket stages are carefully jettisoned during flight to optimize the efficiency of the overall launch.",
    "The concept of a space race between nations fueled significant advancements in rocket technology during the mid-20th century.",
    "Rocketry enthusiasts often participate in amateur rocket launches, showcasing creativity and technical skill.",
    "The Space Launch System's powerful engines generate thrust equivalent to multiple jet engines combined.",
    "Rocket fairings are protective shells that shield payloads from aerodynamic forces during the initial phases of launch.",
    "The study of exoplanets relies on telescopes carried into space by rockets to observe distant planetary systems.",
    "The concept of interplanetary colonization raises ethical and logistical questions related to rocket-based space travel.",
    "In-space refueling technologies are being explored as a means to extend the operational life of satellites and spacecraft.",
    "Rocket design considers factors such as aerodynamics, structural integrity, and thermal management for successful missions.",
    "The concept of a space elevator, if realized, could revolutionize the way payloads are transported from Earth to space.",
    "Rocket launches from different launch sites around the world contribute to the global effort of space exploration.",
    "The study of space weather involves monitoring the effects of solar activity on rockets and satellites in Earth's orbit.",
    "The concept of nuclear thermal propulsion presents a potential solution for faster and more efficient deep-space travel.",
    "Rocket payload adapters are customized to securely attach and deploy various types of payloads during space missions.",
    "The concept of asteroid mining envisions rockets extracting valuable resources from celestial bodies for use on Earth.",
    "Rocket design incorporates redundancy and fail-safes to ensure mission success even in the face of unexpected challenges.",
    "The concept of a spaceplane combines features of both aircraft and rockets for reusable and versatile space travel.",
    "Rocket scientists and engineers collaborate across disciplines to address the multifaceted challenges of space exploration.",
    "The concept of a lunar gateway involves using rockets to establish a human outpost in orbit around the Moon for future missions.",
    "Rocket launches are scheduled with precision, taking into account orbital mechanics and launch windows for optimal trajectories.",
    "The study of microgravity effects on materials and biological systems is conducted aboard rockets during parabolic flights.",
    "Rocket landing technologies, such as those employed by SpaceX's Falcon 9, enable reusable and cost-effective space travel.",
    "The concept of in-situ resource utilization explores the possibility of using resources from other celestial bodies to support space missions.",
    "Rocket scientists play a critical role in advancing our capabilities for exploring the far reaches of our solar system and beyond.",
    "The Artemis program aims to return humans to the Moon, utilizing powerful rockets for lunar exploration.",
    "Rocket trajectories are carefully calculated to ensure precise navigation through the vacuum of space.",
    "The concept of space-based solar power involves using rockets to deploy solar power satellites that beam energy to Earth.",
    "Rocket testing facilities are equipped with specialized infrastructure to simulate the extreme conditions of spaceflight.",
    "The study of space debris and orbital debris mitigation strategies is essential for maintaining the long-term sustainability of space activities.",
    "Rocket propulsion technologies, such as ion drives and solar sails, offer alternative methods for navigating the cosmos.",
    "The concept of a spacewalk involves astronauts using rockets on their spacesuits to maneuver in microgravity outside the spacecraft.",
    "Rocket launch sites, such as Cape Canaveral and Baikonur Cosmodrome, have witnessed historic moments in space exploration.",
    "The concept of a Mars colony envisions using rockets to transport settlers and supplies for the establishment of a human presence on the Red Planet.",
    "Rocket failure investigations contribute to the continuous improvement of safety protocols and engineering practices in the aerospace industry.",
    "The study of planetary defense involves exploring technologies, including rockets, to mitigate potential threats from near-Earth objects.",
    "Rocketry has inspired countless individuals to pursue careers in science, technology, engineering, and mathematics (STEM).",
    "The concept of a space hotel raises questions about the feasibility and logistics of using rockets for commercial space tourism.",
    "Rocket staging events involve the sequential firing and separation of rocket stages to optimize velocity during ascent.",
    "The study of gravitational assists involves using the gravitational pull of celestial bodies to enhance the trajectories and efficiency of rockets.",
    "Rocket payload fairings are aerodynamically shaped to minimize drag during the initial phase of a rocket's ascent.",
    "The concept of a space tug involves using specialized rockets to transport payloads between different orbits or celestial bodies.",
    "Rocket launches contribute to our understanding of Earth's atmosphere, including the dynamics of upper atmospheric layers.",
    "The study of space ecology explores the challenges and solutions related to sustaining life on long-duration space missions launched by rockets.",
    "Rocket-based space telescopes, such as the Hubble Space Telescope, provide invaluable insights into the distant reaches of the universe.",
    "The concept of a space-based manufacturing facility involves using rockets to transport equipment and materials for production in microgravity.",
    "Rocket nozzle designs are optimized to efficiently expel exhaust gases, maximizing thrust during the launch phase.",
    "The study of space law addresses legal considerations related to rocket launches, satellite operations, and international cooperation in space activities.",
    "Rocket trajectories are influenced by celestial bodies, magnetic fields, and other environmental factors encountered during space travel.",
    "The concept of a space capsule involves using rockets to transport crewed or uncrewed vehicles for reentry and landing on Earth.",
    "Rocket stages are often equipped with guidance systems to ensure accurate navigation and trajectory control throughout the launch.",
    "Rocket launches are broadcasted globally, allowing people around the world to witness the awe-inspiring spectacle of space exploration.",
    "The concept of a space colony envisions self-sustaining habitats established using rockets to transport essential resources and equipment.",
    "Rocket-based observations of celestial events, such as eclipses and transits, contribute to scientific understanding and public engagement in astronomy.",
    "The study of space medicine addresses the physiological and psychological challenges faced by astronauts during rocket launches and space missions.",
    "Rocket propellants are selected based on their chemical properties and performance characteristics to meet the specific requirements of each mission.",
    "The concept of a space agency involves national or international organizations coordinating and conducting rocket-based space exploration initiatives.",
    "Rocket testing includes both ground-based tests and in-flight tests to validate the performance and reliability of rocket systems.",
    "Rocket development timelines can span several years, involving research, design, testing, and iterative improvements to achieve mission success.",
    "The concept of space habitats explores the design and construction of living environments transported by rockets for long-term human presence in space.",
    "Rocket payloads, such as scientific instruments and communication satellites, contribute to advancements in technology and our understanding of the universe.",
    "Rocket launches serve as milestones in space history, marking achievements in science, exploration, and international collaboration.",
    "The study of space archaeology explores the remnants of past rocket launches and space activities to understand the cultural and historical significance of space exploration.",
    "Rocket engine technologies, including liquid propulsion and solid rocket boosters, are continuously refined to enhance performance and efficiency.",
    "The concept of space art involves using rockets to deploy art installations or projects that engage with the unique environment of outer space.",
    "Rocket launches from private space companies, in addition to government agencies, contribute to the growing accessibility of space for scientific, commercial, and educational purposes.",
    "The study of space psychology delves into the psychological effects of long-duration space travel, including the mental well-being of astronauts launched by rockets.",
    "Rocket payloads designed for space science experiments contribute to our understanding of fundamental physical principles and phenomena in the cosmos.",
    "The concept of a space concert envisions performances transmitted via rockets to space, reaching audiences beyond the confines of Earth.",
    "Rocket-based exploration of icy moons, such as Europa and Enceladus, aims to uncover potential habitable environments and signs of extraterrestrial life.",
    "Rocket launches are celebrated events that inspire wonder and curiosity, fostering a sense of shared humanity and collective aspirations for the future of space exploration."
]


# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(rocket_data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in rocket_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=2)

# Function to generate text using the trained model
def generate_text(seed_text, next_words, model, max_sequence_length, tokenizer, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0).flatten()

        # Adjust the temperature to control randomness
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)

        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate text using the trained model
generated_text = generate_text("Rockets", 100, model, max_sequence_length, tokenizer, temperature=0.5)
print(generated_text)



