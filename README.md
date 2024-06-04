# Formula SAE PDF Data Extraction Tool

This tool is designed to facilitate the filtering of information from PDF documents related to the Formula SAE competition. Formula SAE is a competition where teams design and build formula-style race cars, which can be either electric or combustion-powered.

## Introduction

The Formula SAE competition involves numerous rules and regulations that teams must comply with during the design and construction of their vehicles. These rules are often documented in PDF files, which can make extracting and organizing information a complicated task. This tool uses Gemini's LLM with Retrieval-Augmented Generation (RAG) as context to extract relevant data from Formula SAE PDF documents.

## ChatBot
This code includes an integration with Flask, allowing users to interact with a ChatBot powered by the model. In the chat, users can view answers to their questions, along with the context that the language model (LLM) used to formulate the responses. The goal is to maintain clarity of information extracted.

## Aditional information
- Set up your Gemini's API key
- The current vector database is set to electric car rules, but you can choose combustion's rules if needed
- Customize the temperature and the number of subtopics used as context for the LLM


_Created by Bruno Finardi Hime_
