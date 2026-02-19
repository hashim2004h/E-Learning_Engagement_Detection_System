document.addEventListener('DOMContentLoaded', () => {
  // ---------- Popup Logic ----------
  function showPopup(message) {
    // If you want a site-wide popup, you can implement it here
    // For example, create a dynamic Bootstrap modal or an overlay.
    alert(message); // quick fallback—replace with a nicer UI if desired
  }

  // ---------- New Meeting ----------
  const newMeetingBtn = document.getElementById('createMeeting');
  if (newMeetingBtn) {
    newMeetingBtn.addEventListener('click', async () => {
      newMeetingBtn.disabled = true;
      try {
        const res = await fetch('/generate-meeting', { method: 'POST' });
        const data = await res.json();
        if (data.code) {
          showPopup(`New meeting code: ${data.code}`);
          setTimeout(() => window.location.reload(), 1500);
        }
      } catch (err) {
        console.error(err);
      }
      newMeetingBtn.disabled = false;
    });
  }

  // ---------- Start/Stop & PDF Buttons ----------
  document.querySelectorAll('.startMeeting').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.dataset.code;
      showPopup(`Starting meeting ${code}…`);
      fetch('/start-meeting', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      }).then(() => btn.disabled = true);
    });
  });
  document.querySelectorAll('.stopMeeting').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.dataset.code;
      showPopup(`Ending meeting ${code}…`);
      fetch('/stop-meeting', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      }).then(res => res.json()).then(data => {
        if (data.success) {
          const card = btn.closest('.col-lg-6');
          if (card) card.remove();
        }
      });
    });
  });
  document.querySelectorAll('.report-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.dataset.code;
      showPopup('Preparing PDF…');
      window.location.href = `/export-pdf/${code}`;
    });
  });

  // ---------- Chart.js Initialization (if on the analytics tab) ----------
  document.querySelectorAll('[id^="engagementChart-"]').forEach(canvas => {
    const meetingCode = canvas.id.replace('engagementChart-', '');
    fetch(`/api/meeting-engagement/${meetingCode}`)
      .then(r => r.json())
      .then(data => {
        new Chart(canvas.getContext('2d'), {
          type: 'line',
          data: {
            labels: data.timestamps,
            datasets: [{
              label: 'Engagement %',
              data: data.engagement_percent,
              borderColor: 'rgba(40, 167, 69, 1)',
              backgroundColor: 'rgba(40, 167, 69, 0.2)',
              fill: true,
              tension: 0.3,
              pointRadius: 3
            }]
          },
          options: {
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                title: { display: true, text: 'Engagement (%)' }
              },
              x: {
                title: { display: true, text: 'Time (HH:MM)' }
              }
            },
            plugins: {
              legend: { display: true, position: 'top' }
            }
          }
        });
      })
      .catch(err => console.error('Chart load error:', err));
  });

  // ---------- Socket.IO (if needed) ----------
  if (window.io) {
    const socket = io();

    // Example: listen for participant updates
    socket.on('participant_update', data => {
      const list = document.getElementById('participant-list');
      if (!list) return;
      list.innerHTML = '';
      data.participants.forEach(name => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = name;
        list.appendChild(li);
      });
    });

    // Example: chat_message
    socket.on('chat_message', msg => {
      const chatWindow = document.getElementById('chat-window');
      if (!chatWindow) return;
      const div = document.createElement('div');
      div.innerHTML = `<strong>${msg.sender}:</strong> ${msg.message}`;
      chatWindow.appendChild(div);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
  }
});
