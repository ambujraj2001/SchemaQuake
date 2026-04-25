# TravelWorld Booking Policies

**Document version:** v1
**Last updated:** 2026-04-15

## 1. Cancellation

- **Refundable** flights and hotels may be cancelled **up to 24 hours** before check-in
  or departure for a full refund.
- **Non-refundable** bookings cannot be cancelled for a refund under any circumstances.
- Refunds are processed in the original currency within 5 business days.

## 2. Pricing

- All prices are quoted in **Indian Rupees (INR)**, using whole rupees (not paise).
- The `price` field on a flight and the `price_per_night` field on a hotel are the
  canonical totals before taxes.
- Taxes are flat 12% and are automatically applied at booking time.

## 3. Refundability field

- The `refundable` field on any flight or hotel is a boolean: `true` means the
  booking can be cancelled for a refund (subject to section 1); `false` means it
  cannot.

## 4. Booking confirmation

- A booking is only confirmed once the `book` tool returns `status: "confirmed"`.
- Customers must receive a confirmation ID (starts with `CNF-`) for every booking.

## 5. User intent priority

- If the user explicitly requests a **refundable** booking, non-refundable options
  must never be chosen, even if they are cheaper.
- If the user sets a **maximum price**, the total (pre-tax) must not exceed it.
